#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
@author: WEW
@contact: wangerwei@tju.edu.cn
@ copy from mmdetection
"""

import warnings
import numpy as np
import torch.nn as nn

# 添加权重的常数初始化，以及对应偏置的常数初始化
def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution=='uniform':
            nn.init.xavier_uniform(module.weight, gain=gain)
        else:
            nn.init.xavier_normal(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def uniform_init(module, a=0, b=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class BaseInit(object):

    def __init__(self, *, bias=0, bias_prob=None, layer=None):
        self.wholemodule = False
        if not isinstance(bias, (int, float)):
            raise TypeError(f'bias must be a numbel, but got a {type(bias)}')

        if bias_prob is not None:
            if not isinstance(bias_prob, float):
                raise TypeError(f'bias_prob type must be float, \
                    but got {type(bias_prob)}')

        if layer is not None:
            if not isinstance(layer, (str, list)):
                raise TypeError(f'layer must be a str or a list of str, \
                    but got a {type(layer)}')
        else:
            layer = []
            warnings.warn(
                'init_cfg without layer key, if you do not define override'
                ' key either, this init_cfg will do nothing')
        if bias_prob is not None:
            self.bias = bias_init_with_prob(bias_prob)
        else:
            self.bias = bias
        self.layer = [layer] if isinstance(layer, str) else layer



class ConstantInit(BaseInit):
    """Initialize module parameters with constant values.
    Args:
        val (int | float): the value to fill the weights in the module with
        bias (int | float): the value to fill the bias or
        define initialization type for bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, val, **kwargs):
        super().__init__(**kwargs)
        self.val = val

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                constant_init(m, self.val, self.bias)
            else:
                layername = m.__class__.__name__
                if layername in self.layer:
                    constant_init(m, self.val, self.bias)

        module.apply(init)

class XavierInit(BaseInit):
    r"""Initialize module parameters with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks - Glorot, X. & Bengio, Y. (2010).
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    Args:
        gain (int | float): an optional scaling factor. Defaults to 1.
        bias (int | float): the value to fill the bias or define
            initialization type for bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'``
            or ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, gain=1, distribution='normal', **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                xavier_init(m, self.gain, self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                if layername in self.layer:
                    xavier_init(m, self.gain, self.bias, self.distribution)

        module.apply(init)


class NormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.
    Args:
        mean (int | float):the mean of the normal distribution. Defaults to 0.
        std (int | float): the standard deviation of the normal distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias or define
            initialization type for bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, mean=0, std=1, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                normal_init(m, self.mean, self.std, self.bias)
            else:
                layername = m.__class__.__name__
                for layer_ in self.layer:
                    if layername == layer_:
                        normal_init(m, self.mean, self.std, self.bias)

        module.apply(init)

class UniformInit(BaseInit):
    r"""Initialize module parameters with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        a (int | float): the lower bound of the uniform distribution.
            Defaults to 0.
        b (int | float): the upper bound of the uniform distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias or define
            initialization type for bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, a=0, b=1, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                uniform_init(m, self.a, self.b, self.bias)
            else:
                layername = m.__class__.__name__
                if layername in self.layer:
                    uniform_init(m, self.a, self.b, self.bias)

        module.apply(init)

class KaimingInit(BaseInit):
    r"""Initialize module paramters with the valuse according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification - He, K. et al. (2015).
    <https://www.cv-foundation.org/openaccess/content_iccv_2015/
    papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`_
    Args:
        a (int | float): the negative slope of the rectifier used after this
            layer (only used with ``'leaky_relu'``). Defaults to 0.
        mode (str):  either ``'fan_in'`` or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the
            magnitudes in the backwards pass. Defaults to ``'fan_out'``.
        nonlinearity (str): the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` .
            Defaults to 'relu'.
        bias (int | float): the value to fill the bias or define
            initialization type for bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'`` or
            ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 distribution='normal',
                 **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                kaiming_init(m, self.a, self.mode, self.nonlinearity,
                             self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                if layername in self.layer:
                    kaiming_init(m, self.a, self.mode, self.nonlinearity,
                                 self.bias, self.distribution)

        module.apply(init)