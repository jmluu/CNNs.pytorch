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

    def __init__(self, index, num_class, anchor, stride, 
            alloc_size=(128, 128), **kwargs):
            super(YOLOOutputV3, self).__init__(**kwargs)
        
        self._classes = num_class
        self._num_pred = 1 + 4 + num_class  # 1 objness + 4 box + num_class
        self._num_anchors = anchors.size // 2
        self._stride = stride



