#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
@author: WEW
@contact: wangerwei@tju.edu.cn
"""

import torch.nn as nn


#卷积、bn、relu一体的函数
class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1, relu=nn.ReLU):
        super(Conv_bn_relu, self).__init__()
        self.relu = relu
        if self.relu is not None:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                self.relu(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.convs(x)
        return out

# 卷积核为3的卷积
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# 卷积核为1的卷积
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)