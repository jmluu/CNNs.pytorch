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
    

class Upsample(nn.Module):

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        output = F.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)

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
        raw_box_centers = pred[:, :, :, 0 : 2]  # center x, y
        raw_box_scales = pred[:, :, :, 2 : 4]   # height  and width scale
        objness = pred[:, :, :, 4]              # confidence;
        class_pred = pred[:, :, :, 5]

        # valid offsets, (1, 1, height, width, 2)
        offsets = offsets[:, :, :height, :width, :]
        # reshape to (1, height*width, 1, 2)
        offsets = offsets.view(1, -1, 1, 2)

        box_centers = (torch.sigmoid(raw_box_centers) + offsets) * self._stride
        box_scales = torch.exp(raw_box_scales) * anchors
        confidence = torch.sigmoid(objness)
        class_score = torch.sigmoid(class_pred) * confidence
        wh = box_scales / 2.0 
        bbox = torch.cat(
            (
                box_centers - wh, box_centers + wh 
            ), -1 )

        if self.training : 
            return (bbox.view(0, -1, 4), raw_box_centers, raw_box_scales, 
                objness, class_pred, anchors, offsets)

        
        # prediction per class 
        bboxes = bbox.repeat(self._classes, 1, 1, 1, 1)
        scores = class_score.permute(3, 0, 1, 2).unsqueeze(-1)
        ids = scores.mul(0) + torch.arange(0, self._classes).view(-1, 1, 1, 1, 1)
        detections = torch.cat((ids, scores, bboxes), -1 )

        # reshape to (B, xx, 6)
        detections = detections.permute(1,0,2,3,4).view(batch, -1, 6 ).contiguous()

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


