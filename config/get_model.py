#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================
from model.losses import *
from model.head import *
from model.maskfeat_head import *
from model.resnet_vd import *
from model.resnet_vb import *
from model.fpn import *


def select_backbone(name):
    if name == 'Resnet50Vd':
        return Resnet50Vd
    if name == 'Resnet18Vd':
        return Resnet18Vd
    if name == 'Resnet50Vb':
        return Resnet50Vb

def select_head(name):
    if name == 'SOLOv2Head':
        return SOLOv2Head
    if name == 'MaskFeatHead':
        return MaskFeatHead

def select_fpn(name):
    if name == 'FPN':
        return FPN

def select_loss(name):
    if name == 'SOLOv2Loss':
        return SOLOv2Loss

def select_regularization(name):
    if name == 'L1Decay':
        return fluid.regularizer.L1Decay
    if name == 'L2Decay':
        return fluid.regularizer.L2Decay

def select_optimizer(name):
    if name == 'Momentum':
        return paddle.optimizer.Momentum
    if name == 'Adam':
        return paddle.optimizer.Adam
    if name == 'SGD':
        return paddle.optimizer.SGD




