"""Transforms for YOLO series."""
from __future__ import absolute_import
import copy
import torch
import torch.nn as nn 
import numpy as np
import torchvision.transforms.functional as F 

import bbox as tbbox 

__all__ = ['transform_test', 'load_test', 'YOLO3DefaultTrainTransform', 'YOLO3DefaultValTransform']


class YOLO3DefaultTrainTransform(object):
    """Default YOLO training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    net : torch.nn.Module, optional
        The yolo network.

        .. hint::

            If net is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, width, height, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), mixup=False, **kwargs):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._mixup = mixup
        self._target_generator = None
        if net is None:
            return

        

        # TODO : add Train Transform (need YOLOV3PrefetchTargetGenerator, and Net )
        # generate fake image and lable;
        self._fake_x = 

class YOLO3DefaultValTransform(object):
    """Default YOLO validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        """Apply transform to validation image/label.
            src (PIL Image): Image to be resized: 
        """
        # resize
        h, w, _ = src.size 
        img = F.resize(src, (self._width, self._height), interpolation=9)
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = F.to_tensor(img)
        img = F.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype(img.dtype)