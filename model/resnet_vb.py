#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import paddle
import paddle.nn.functional as F
from model.custom_layers import *
from model.resnet_vd import IdentityBlock


class ConvBlock(paddle.nn.Layer):
    def __init__(self, in_c, filters, bn, gn, af, freeze_norm, norm_decay, lr, use_dcn=False, stride=2, downsample_in3x3=True, block_name=''):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        if downsample_in3x3 == True:
            stride1, stride2 = 1, stride
        else:
            stride1, stride2 = stride, 1

        self.conv1 = Conv2dUnit(in_c,     filters1, 1, stride=stride1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', name=block_name+'_branch2a')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act='relu', use_dcn=use_dcn, name=block_name+'_branch2b')
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch2c')

        self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=stride, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, lr=lr, act=None, name=block_name+'_branch1')
        self.act = paddle.nn.ReLU()

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.conv3.freeze()
        self.conv4.freeze()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.conv4(input_tensor)
        x = x + shortcut
        x = self.act(x)
        return x


class Resnet50Vb(paddle.nn.Layer):
    def __init__(self, norm_type='bn', feature_maps=[3, 4, 5], dcn_v2_stages=[5], downsample_in3x3=True, freeze_at=0, freeze_norm=False, norm_decay=0., lr_mult_list=[1., 1., 1., 1.]):
        super(Resnet50Vb, self).__init__()
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        assert freeze_at in [0, 1, 2, 3, 4, 5]
        assert len(lr_mult_list) == 4, "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        self.lr_mult_list = lr_mult_list
        self.freeze_at = freeze_at
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        self.stage1_conv1_1 = Conv2dUnit(3,  64, 7, stride=2, bn=bn, gn=gn, af=af, freeze_norm=freeze_norm, norm_decay=norm_decay, act='relu', name='conv1')
        self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # stage2
        self.stage2_0 = ConvBlock(64, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], stride=1, downsample_in3x3=downsample_in3x3, block_name='res2a')
        self.stage2_1 = IdentityBlock(256, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], block_name='res2b')
        self.stage2_2 = IdentityBlock(256, [64, 64, 256], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[0], block_name='res2c')

        # stage3
        use_dcn = 3 in dcn_v2_stages
        self.stage3_0 = ConvBlock(256, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res3a')
        self.stage3_1 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3b')
        self.stage3_2 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3c')
        self.stage3_3 = IdentityBlock(512, [128, 128, 512], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[1], use_dcn=use_dcn, block_name='res3d')

        # stage4
        use_dcn = 4 in dcn_v2_stages
        self.stage4_0 = ConvBlock(512, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res4a')
        self.stage4_1 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4b')
        self.stage4_2 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4c')
        self.stage4_3 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4d')
        self.stage4_4 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4e')
        self.stage4_5 = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[2], use_dcn=use_dcn, block_name='res4f')

        # stage5
        use_dcn = 5 in dcn_v2_stages
        self.stage5_0 = ConvBlock(1024, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, downsample_in3x3=downsample_in3x3, block_name='res5a')
        self.stage5_1 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name='res5b')
        self.stage5_2 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, freeze_norm, norm_decay, lr_mult_list[3], use_dcn=use_dcn, block_name='res5c')

    def forward(self, input_tensor):
        x = self.stage1_conv1_1(input_tensor)
        x = self.pool(x)

        # stage2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
        # stage3
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
        # stage4
        x = self.stage4_0(s8)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        s16 = self.stage4_5(x)
        # stage5
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)

        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 1:
            self.stage1_conv1_1.freeze()
        if freeze_at >= 2:
            self.stage2_0.freeze()
            self.stage2_1.freeze()
            self.stage2_2.freeze()
        if freeze_at >= 3:
            self.stage3_0.freeze()
            self.stage3_1.freeze()
            self.stage3_2.freeze()
            self.stage3_3.freeze()
        if freeze_at >= 4:
            self.stage4_0.freeze()
            self.stage4_1.freeze()
            self.stage4_2.freeze()
            self.stage4_3.freeze()
            self.stage4_4.freeze()
            self.stage4_5.freeze()
        if freeze_at >= 5:
            self.stage5_0.freeze()
            self.stage5_1.freeze()
            self.stage5_2.freeze()






