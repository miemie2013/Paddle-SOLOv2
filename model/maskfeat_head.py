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
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.layers as L

from model.custom_layers import Conv2dUnit, get_norm


class MaskFeatHead(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type,
                 start_level,
                 end_level,
                 num_classes):
        super(MaskFeatHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes

        bn, gn, af = get_norm(norm_type)
        self.convs_all_levels = paddle.nn.LayerList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = paddle.nn.LayerList()
            if i == 0:
                one_conv = Conv2dUnit(self.in_channels, self.out_channels, 3, stride=1,
                                      bias_attr=False, bn=bn, gn=gn, af=af, groups=32, act='relu',
                                      name='mask_feat_head.convs_all_levels.%d.conv0' % (i,))
                convs_per_level.append(one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    one_conv = Conv2dUnit(chn, self.out_channels, 3, stride=1,
                                          bias_attr=False, bn=bn, gn=gn, af=af, groups=32, act='relu',
                                          name='mask_feat_head.convs_all_levels.%d.conv%d' % (i, j))
                    convs_per_level.append(one_conv)
                    one_upsample = paddle.nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False, align_mode=0)
                    convs_per_level.append(one_upsample)
                    continue

                one_conv = Conv2dUnit(self.out_channels, self.out_channels, 3, stride=1,
                                      bias_attr=False, bn=bn, gn=gn, af=af, groups=32, act='relu',
                                      name='mask_feat_head.convs_all_levels.%d.conv%d' % (i, j))
                convs_per_level.append(one_conv)
                one_upsample = paddle.nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False, align_mode=0)
                convs_per_level.append(one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = Conv2dUnit(self.out_channels, self.num_classes, 1, stride=1,
                                    bias_attr=False, bn=bn, gn=gn, af=af, groups=32, act='relu',
                                    name='mask_feat_head.conv_pred.0')

    def forward(self, inputs):
        # [p2, p3, p4, p5, p6] -> [p2, p3, p4, p5]
        inputs = inputs[self.start_level:self.end_level+1]

        feature_add_all_level = self.convs_all_levels[0][0](inputs[0])

        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                batch_size = L.shape(input_feat)[0]
                h = L.shape(input_feat)[2]
                w = L.shape(input_feat)[3]
                float_h = L.cast(h, 'float32')
                float_w = L.cast(w, 'float32')

                y_range = L.range(0., float_h, 1., dtype='float32')  # [h, ]
                y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
                x_range = L.range(0., float_w, 1., dtype='float32')  # [w, ]
                x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
                x_range = L.reshape(x_range, (1, -1))  # [1, w]
                y_range = L.reshape(y_range, (-1, 1))  # [h, 1]
                x = L.expand(x_range, [h, 1])  # [h, w]
                y = L.expand(y_range, [1, w])  # [h, w]

                x = L.reshape(x, (1, 1, h, w))  # [1, 1, h, w]
                y = L.reshape(y, (1, 1, h, w))  # [1, 1, h, w]
                x = L.expand(x, [batch_size, 1, 1, 1])  # [N, 1, h, w]
                y = L.expand(y, [batch_size, 1, 1, 1])  # [N, 1, h, w]

                input_p = L.concat([input_p, x, y], axis=1)  # [N, c+2, h, w]

            for ly in self.convs_all_levels[i]:
                input_p = ly(input_p)
            feature_add_all_level += input_p

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred








