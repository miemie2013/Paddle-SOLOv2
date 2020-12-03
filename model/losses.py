#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      :
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================
import paddle
import paddle.nn.functional as F
import paddle.fluid.layers as L
import numpy as np
import paddle.fluid as fluid

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence



class SOLOv2Loss(object):
    """
    SOLOv2Loss
    Args:
        ins_loss_weight (float): Weight of instance loss.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        focal_loss_alpha (float): Alpha parameter for focal loss.
    """

    def __init__(self,
                 ins_loss_weight=3.0,
                 focal_loss_gamma=2.0,
                 focal_loss_alpha=0.25):
        self.ins_loss_weight = ins_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

    def _dice_loss(self, input, target):
        input = fluid.layers.reshape(
            input, shape=(fluid.layers.shape(input)[0], -1))
        target = fluid.layers.reshape(
            target, shape=(fluid.layers.shape(target)[0], -1))
        target = fluid.layers.cast(target, 'float32')
        a = fluid.layers.reduce_sum(input * target, dim=1)
        b = fluid.layers.reduce_sum(input * input, dim=1) + 0.001
        c = fluid.layers.reduce_sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def __call__(self, ins_pred_list, ins_label_list, cate_preds, cate_labels,
                 num_ins):
        """
        Get loss of network of SOLOv2.
        Args:
            ins_pred_list (list):  接住例子。 第0个元素 [5, s4, s4] 。是预测的掩码，未进行sigmoid()激活。
            ins_label_list (list):  5个元素。5个输出层的正样本对应掩码。里面每个元素形状是[M, s4, s4]  M表示该输出层所有图片的正样本对应掩码 被无差别地拼接起来。从 小感受野 到 大感受野。
            cate_preds (list): Concat Variable list of categroy branch output.
            cate_labels (list): Concat list of categroy labels pre batch.
            num_ins (int): Number of positive samples in a mini-batch.
        Returns:
            loss_ins (Variable): The instance loss Variable of SOLOv2 network.
            loss_cate (Variable): The category loss Variable of SOLOv2 network.
        """

        # Ues dice_loss to calculate instance loss
        loss_ins = []
        total_weights = fluid.layers.zeros(shape=[1], dtype='float32')   # 正样本个数

        # 遍历每个输出层
        for input, target in zip(ins_pred_list, ins_label_list):
            # input.shape=[5, s4, s4]   是预测的掩码，未进行sigmoid()激活。
            # target.shape=[M=5, s4, s4]   是真实的掩码。

            # 假如是“假掩码”(预处理时假如该层没正样本，就会分配1个全0掩码)，权重为0，不计算该样本(负样本)的掩码损失。
            weights = L.cast(L.reduce_sum(target, dim=[1, 2]) > 0, 'float32')   # [M, ]
            input = fluid.layers.sigmoid(input)
            dice_out = fluid.layers.elementwise_mul(
                self._dice_loss(input, target), weights)   # 不计算假正样本(负样本)的掩码损失
            total_weights += fluid.layers.reduce_sum(weights)
            loss_ins.append(dice_out)
        loss_ins = fluid.layers.reduce_sum(fluid.layers.concat(
            loss_ins)) / total_weights
        loss_ins = loss_ins * self.ins_loss_weight

        # Ues sigmoid_focal_loss to calculate category loss
        # 使用sigmoid_focal_loss计算分类损失
        # 类别id(cate_labels) 取值范围是[0, 80]共81个值。类别id是0时表示的是背景。
        loss_cate = fluid.layers.sigmoid_focal_loss(
            x=cate_preds,
            label=cate_labels,
            fg_num=num_ins + 1,
            gamma=self.focal_loss_gamma,
            alpha=self.focal_loss_alpha)
        # sigmoid_focal_loss的一种等价实现。
        # loss_cate = self.sigmoid_focal_loss(
        #     x=cate_preds,
        #     label=cate_labels,
        #     fg_num=num_ins + 1,
        #     gamma=self.focal_loss_gamma,
        #     alpha=self.focal_loss_alpha)
        loss_cate = fluid.layers.reduce_sum(loss_cate)

        return loss_ins, loss_cate


    def sigmoid_focal_loss(self, x, label, fg_num, gamma=2.0, alpha=0.25):
        C = x.shape[1]
        eye = paddle.eye(C + 1, dtype='float32')
        one_hot = L.gather(eye, label)
        pos_mask = one_hot[:, 1:]  # 正样本掩码

        p = L.sigmoid(x)  # [批大小*所有格子数, 80]， 预测的类别概率
        pos_loss = pos_mask * (0 - L.log(p + 1e-9)) * L.pow(1 - p, gamma) * alpha
        neg_loss = (1.0 - pos_mask) * (0 - L.log(1 - p + 1e-9)) * L.pow(p, gamma) * (1 - alpha)
        focal_loss = pos_loss + neg_loss
        if fg_num > 0.5:   # 当没有gt时，即fg_num==0时，focal_loss什么都不除。
            focal_loss = focal_loss / fg_num
        return focal_loss




