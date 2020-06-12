"""You Only Look Once Object Detection v3"""
# pylint: disable=arguments-differ
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as base_models


def _conv2d(channel, kernel_size, padding, stride):
    cell = nn.Sequential(
        nn.Conv2d(channel, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(channel), 
        nn.LeakyReLU(0.1)
        )

    return cell 
    

def _upsample(input, scale_factor):

    output = F.interpolate(input, scale_factor=scale_factor, mode='nearest')

    return output 

class YOLOOutputV3(nn.Module):
    """
    YOLO output layer V3.
    Parameters
    ----------
    index : int
        Index of the yolo output layer, to avoid naming conflicts only.
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. Reference: https://arxiv.org/pdf/1804.02767.pdf.
    stride : int
        Stride of feature map.
    alloc_size : tuple of int, default is (128, 128)
    """

    def __init__(self, index, num_class, anchors, stride, 
            alloc_size=(128, 128), **kwargs):
            super(YOLOOutputV3, self).__init__(**kwargs)

        self._classes = num_class
        self._num_pred = 1 + 4 + num_class  # 1 objness + 4 box + num_class
        self._num_anchors = anchors.size // 2
        self._stride = stride

        all_pred = self._num_pred * self._num_anchors
        self.prediction = nn.Conv2d(all_pred, kernel_size=1, padding=0, strides=1)
        
        # anchors will be multiplied to predictions
        anchors = anchors.reshape(1, 1, -1, 2)
        self.anchors = self.register_buffer('anchor_%d'%(index), anchors)
        
        # offsets will be added to predictions
        grid_x = np.arange(alloc_size[1])
        grid_y = np.arange(alloc_size[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        # stack to (n, n, 2)
        offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
        # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
        offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
        self.offsets = self.register_buffer('offset_%d'%(index), offsets)

    def reset_class(self, classes, reuse_weight=None):
        NotImplemented
        #TODO: Implemente transfer feature;

    def forward(self, x, anchors, offsets):
        batch = x.shape[0]
        height= x.shape[2]
        width = x.shape[3]

        # predictoin flat to (batch, pred per pixel, height, * width)      
        pred = self.prediction(x).view(batch, self._num_anchors * self._num_pred, -1)
        # transpose to (batch, height * width, num_anchor, num_pred)
        pred = pred.permute(0, 2, 1).view(batch, -1, self._num_anchors, self._num_pred).contiguous()

        # components 
        raw_box_centers = pred[:, :, :, 0 : 2]  # center x, y               [batch, height * width, num_anchor, 2]
        raw_box_scales = pred[:, :, :, 2 : 4]   # height  and width scale   [batch, height * width, num_anchor, 2]
        objness = pred[:, :, :, 4]              # confidence;               [batch, height * width, num_anchor, 1]
        class_pred = pred[:, :, :, 5 : ]        # class                     [batch, height * width, num_anchor, 80]

        # valid offsets, (1, 1, height, width, 2)
        offsets = offsets[:, :, :height, :width, :]
        # reshape to (1, height*width, 1, 2)
        offsets = offsets.view(1, -1, 1, 2)

        box_centers = (torch.sigmoid(raw_box_centers) + offsets) * self._stride # center x, y               [batch, height * width, num_anchor, 2]
        box_scales = torch.exp(raw_box_scales) * anchors                        # height  and width scale   [batch, height * width, num_anchor, 2]
        confidence = torch.sigmoid(objness)                                     # confidence;               [batch, height * width, num_anchor, 1]    
        class_score = torch.sigmoid(class_pred) * confidence                    # class                     [batch, height * width, num_anchor, 80]    
        wh = box_scales / 2.0 
        bbox = torch.cat(
            (
                box_centers - wh, box_centers + wh 
            ), -1 )                                                             # box                       [batch, height * width, num_anchor, 4]

        if self.training : 
            return (bbox.view(0, -1, 4), raw_box_centers, raw_box_scales, 
                objness, class_pred, anchors, offsets)

        
        # prediction per class 
        bboxes = bbox.repeat(self._classes, 1, 1, 1, 1)                         # boxes                      [80, batch, height * width, num_anchor, 4]
        scores = class_score.permute(3, 0, 1, 2).unsqueeze(-1)                  # class                      [80, batch, height * width, num_anchor, 1] 
        ids = scores.mul(0) + torch.arange(0, self._classes).view(-1, 1, 1, 1, 1)# ids                       [80, batch, height * width, num_anchor, 1] 
        detections = torch.cat((ids, scores, bboxes), -1 )

        # reshape to (B, xx, 6)
        detections = detections.permute(1,0,2,3,4).view(batch, -1, 6 ).contiguous()  # resuit        [batch, height * width * num_anchor * num_class, 6] 

        return detections

class YOLODetectionBlockV3(nn.Module):

    def __init__(self, channel, **kwargs):
        super(YOLODetectionBlockV3, self).__init__(**kwargs)
        assert channel % 2 == 0

        self.body = nn.Sequential()

        for i in range(2): 
            self.body.add_module("%d.0"%i, _conv2d(channel, 1, 0, 1))
            self.body.add_module("%d.1"%i, _conv2d(channel*2, 3, 1, 1))
        
        self.body.add_module(_conv2d(channel, 1, 0, 1))

        self.tip = _conv2d(channel * 2, 3, 1, 1)

    def forward(self, input):
        route = self.body(x)
        tip = self.tip(route)

        return route, tip



class   YOLOV3(nn.Module):

    def __init__(self, stages, channels, anchors, strides, classes, alloc_size=(128, 128),
                 nms_thresh=0.45, nms_topk=400, post_nms=100, pos_iou_thresh=1.0,
                 ignore_iou_thresh=0.7, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(YOLOV3, self).__init__(**kwargs)

        self._classes = classes
        self._num_class = len(classes)
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self._pos_iou_thresh = pos_iou_thresh
        self._ignore_iou_thresh = ignore_iou_thresh
        if pos_iou_thresh >= 1:
            # TODO: add YOLOV3TargetMerger
            self._target_generator = YOLOV3TargetMerger(len(classes), ignore_iou_thresh)
        else:
            raise NotImplementedError(
                "pos_iou_thresh({}) < 1.0 is not implemented!".format(pos_iou_thresh))

        self._loss = YOLOV3Loss()
        self.make_layer()

    def make_layer(self, stages, channels, anchors, strides): 

        self.stages = nn.Sequential()
        self.transitions = nn.Sequential()
        self.yolo_blocks = nn.Sequential()
        self.yolo_outputs = nn.Sequential()

        for i, stage, channel, anchor, stride in zip(
            range(len(stages)), stages, channels, anchors[::-1], strides[::-1]):

            self.stages.add_module('stage_%d'%i, stage), 
            self.yolo_blocks.add_module( 'yolo_detect_d'%i,YOLODetectionBlockV3(channel))
            self.yolo_outputs.add_module('yolo_output_d'%i, 
                YOLOOutputV3(i, self._num_class, anchor, stride, alloc_size))

            if i > 0 : 
                self.transitions.add_module('transition_%d'%i, 
                    _conv2d(channel, 1, 0, 1))

    def forward(self, input):

        all_box_centers = []
        all_box_scales = []
        all_objectness = []
        all_class_pred = []
        all_anchors = []
        all_offsets = []
        all_feat_maps = []
        all_detections = []
        routes = []

        batch = input.shape[0]


        for stage, block, output in zip(self.stages, self.yolo_blocks, self.yolo_outputs):
            x = stage(x)
            routes.append(x)

        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            x, tip = block(x)

            if self.training: 
                dets, box_centers, box_scales, objness, class_pred, anchors, offsets = output(tip)

                num_boxes = box_centers.shape[1] * box_centers.shape[2]
                all_box_centers.append(box_centers.view(batch, num_boxes, -1))
                all_box_scales.append(box_scales.view(batch, num_boxes, -1))
                all_objectness.append(objness.view(batch, num_boxes, -1))
                all_class_pred.append(class_pred.view(batch, num_boxes, -1))
                all_anchors.append(anchors)
                all_offsets.append(offsets)

                fake_featmap = torch.zeros_like(tip[0, 0])
                all_feat_maps.append(fake_featmap)

            else :
                dets = output(tip)
            
            all_detections.append(dets)
            if i >= len(routes) - 1 :
                break

            # add trasition layers
            x = self.transitions[i](x)

            # upsample to shallow layers 
            upsample = _upsample(x, stride=2)
            route_now = routes[::-1][i+1]

            x = torch.cat( (upsample[:,:, :route_now.shape[2], :route_now.shape[3]], route_now), dim = 1 )

        if self.training:
            # during training, the network behaves differently since we don't need detection results
            
            # generate losses and return them directly
            box_preds = torch.cat(all_detections, dim=1)
            all_preds = [torch.cat(p, dim=1) for p in [
                all_objectness, all_box_centers, all_box_scales, all_class_pred]]

            # TODO: target-generator
            all_targets = self._target_generator(box_preds, *args)
            return self._loss(*(all_preds + all_targets))

            # TODO: transform interface; return raw predictions, this is only used in DataLoader transform function.
            # return (F.concat(*all_detections, dim=1), all_anchors, all_offsets, all_feat_maps,
            #         F.concat(*all_box_centers, dim=1), F.concat(*all_box_scales, dim=1),
            #         F.concat(*all_objectness, dim=1), F.concat(*all_class_pred, dim=1))

        # concat all detection results from different stages
        result = torch.cat(all_detections, dim=1)

        # apply nms per class
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            # TODO： add nms 
            result = box_nms(
                result, overlap_thresh=self.nms_thresh, valid_thresh=0.01,
                topk=self.nms_topk, id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result[:, :self.post_nms] 
        ids = result.slice_axis(axis=-1, begin=0, end=1)
        scores = result.slice_axis(axis=-1, begin=1, end=2)
        bboxes = result.slice_axis(axis=-1, begin=2, end=None)
        return ids, scores, bboxes

