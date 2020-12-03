#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================
import paddle


class SOLOv2(paddle.nn.Layer):
    def __init__(self, backbone, fpn, mask_feat_head, head):
        super(SOLOv2, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.mask_feat_head = mask_feat_head
        self.head = head

    def forward(self, x, ori_shape, resize_shape):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.fpn(body_feats)

        # MaskFeatHead。 [bs, 256, s4, s4]   掩码原型
        mask_feats = self.mask_feat_head(body_feats)

        # kernel_preds里每个元素形状是[N, 256, seg_num_grid, seg_num_grid],  每个格子的预测卷积核。      从 小感受野 到 大感受野。
        # cls_preds里每个元素形状是   [N, seg_num_grid, seg_num_grid,  80],  每个格子的预测概率，已进行sigmoid()激活。  从 小感受野 到 大感受野。
        kernel_preds, cls_preds = self.head.get_prediction(body_feats, eval=True)
        pred = self.head.get_seg(kernel_preds, cls_preds, mask_feats, ori_shape, resize_shape)
        return pred

    def train_model(self, x, ins_labels, cate_labels, grid_orders, fg_nums):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.fpn(body_feats)

        # MaskFeatHead。 [bs, 256, s4, s4]   掩码原型
        mask_feats = self.mask_feat_head(body_feats)

        # kernel_preds里每个元素形状是[N, 256, seg_num_grid, seg_num_grid],  每个格子的预测卷积核。      从 小感受野 到 大感受野。
        # cls_preds里每个元素形状是   [N,  80, seg_num_grid, seg_num_grid],  每个格子的预测概率，未进行sigmoid()激活。  从 小感受野 到 大感受野。
        kernel_preds, cls_preds = self.head.get_prediction(body_feats, eval=False)
        out = self.head.get_loss(kernel_preds, cls_preds, mask_feats, ins_labels, cate_labels, grid_orders, fg_nums)
        return out



