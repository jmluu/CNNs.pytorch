from __future__ import division
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import json


__all__ = ['CIFARResNet', 'cifar_resnet20', 'cifar_resnet56', 'cifar_resnet110']




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=False,
                  norm_layer=None):
        super(CIFARBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = norm_layer(planes)
        if downsample :
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,bias=False),
                norm_layer(planes)
            )
        else :
            self.downsample = None


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample :
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return  out

class CIFARResNet(nn.Module):

    def __init__(self, block, layers, planes, num_classes=10, norm_layer=nn.BatchNorm2d):
        super(CIFARResNet, self).__init__()
        assert len(layers) == len(planes)-1
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)

        features = []
        for i, num_layer in enumerate(layers):
            stride = 1 if i==0 else 2
            features.append(self._make_layer(block, planes[i], planes[i+1],
                                             num_layer, stride=stride))
        features.append(nn.AdaptiveAvgPool2d((1,1)))

        self.feature = nn.Sequential(*features)
        self.fc = nn.Linear(planes[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer

        if inplanes != planes * block.expansion:
            downsample = True
        else :
            downsample = False

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample,
                            norm_layer=norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.feature(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _get_resnet_spec(num_layers):
    assert (num_layers - 2) % 6 == 0

    n = (num_layers - 2) // 6
    channels = [16, 16, 32, 64]
    layers = [n] * (len(channels) - 1)
    return layers, channels

def _cifar_resnet(num_layers, **kwargs):
    layers, planes = _get_resnet_spec(num_layers)

    model = CIFARResNet(CIFARBasicBlock, layers, planes, **kwargs)

    return model

def cifar_resnet20(**kwargs):
    return _cifar_resnet(20, **kwargs)

def cifar_resnet56(**kwargs):
    return _cifar_resnet(56, **kwargs)

def cifar_resnet110(**kwargs):
    return _cifar_resnet(110, **kwargs)



if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)

    model = cifar_resnet20()

    y = model(x)
    print(y.shape)
