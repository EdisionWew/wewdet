#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
@author: WEW
@contact: wangerwei@tju.edu.cn
"""
import warnings

def build_backbone(cfg):
    """Build backbone."""
    pass


def build_neck(cfg):
    """Build neck."""
    pass


def build_roi_extractor(cfg):
    """Build roi extractor."""
    pass


def build_shared_head(cfg):
    """Build shared head."""
    pass


def build_head(cfg):
    """Build head."""
    pass


def build_loss(cfg):
    """Build loss."""
    pass


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    pass