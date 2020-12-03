import copy
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay


import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.layers as L

from model.custom_layers import *


class FPN(paddle.nn.Layer):
    """
    Feature Pyramid Network, see https://arxiv.org/abs/1612.03144

    Args:
        num_chan (int): number of feature channels
        min_level (int): lowest level of the backbone feature map to use
        max_level (int): highest level of the backbone feature map to use
        spatial_scale (list): feature map scaling factor
        has_extra_convs (bool): whether has extral convolutions in higher levels
        norm_type (str|None): normalization type, 'bn'/'sync_bn'/'affine_channel'
        norm_decay (float): weight decay for normalization layer weights.
        reverse_out (bool): whether to flip the output.
    """
    __shared__ = ['norm_type', 'freeze_norm']

    def __init__(self,
                 in_channels=[2048, 1024, 512, 256],
                 num_chan=256,
                 min_level=2,
                 max_level=6,
                 spatial_scale=[1. / 32., 1. / 16., 1. / 8., 1. / 4.],
                 has_extra_convs=False,
                 norm_type=None,
                 norm_decay=0.,
                 freeze_norm=False,
                 use_c5=True,
                 reverse_out=False):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.freeze_norm = freeze_norm
        self.num_chan = num_chan
        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale
        self.has_extra_convs = has_extra_convs
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel', None]
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.use_c5 = use_c5
        self.reverse_out = reverse_out

        self.num_backbone_stages = len(in_channels)   # 进入FPN的张量个数
        self.fpn_inner_convs = paddle.nn.LayerList()  # 骨干网络的张量s32, s16, s8, ...使用的卷积
        self.fpn_convs = paddle.nn.LayerList()        # fs32, fs16, fs8, ...使用的卷积

        # fpn_inner_convs
        for i in range(0, self.num_backbone_stages):
            cname = 'fpn_inner_res%d_sum_lateral' % (5 - i, )
            if i == 0:
                cname = 'fpn_inner_res%d_sum' % (5 - i, )
            bn, gn, af = get_norm(norm_type)
            use_bias = True if (bn + gn + af) == 0 else False
            conv = Conv2dUnit(in_channels[i], self.num_chan, 1, stride=1, bias_attr=use_bias, bn=bn, gn=gn, af=af, bias_lr=2.0,
                              weight_init=Xavier(fan_out=in_channels[i] * 1 * 1), bias_init=Constant(0.0),
                              act=None, freeze_norm=self.freeze_norm, norm_decay=self.norm_decay, name=cname)
            self.fpn_inner_convs.append(conv)

        # fpn_convs
        for i in range(0, self.num_backbone_stages):
            bn, gn, af = get_norm(norm_type)
            use_bias = True if (bn + gn + af) == 0 else False
            conv = Conv2dUnit(self.num_chan, self.num_chan, 3, stride=1, bias_attr=use_bias, bn=bn, gn=gn, af=af, bias_lr=2.0,
                              weight_init=Xavier(fan_out=self.num_chan * 3 * 3), bias_init=Constant(0.0),
                              act=None, freeze_norm=self.freeze_norm, norm_decay=self.norm_decay, name='fpn_res%d_sum' % (5 - i, ))
            self.fpn_convs.append(conv)

        # 生成其它尺度的特征图时如果用的是池化层
        # pass

        # 生成其它尺度的特征图时如果用的是卷积层
        self.extra_convs = None
        highest_backbone_level = self.min_level + len(spatial_scale) - 1
        if self.has_extra_convs and self.max_level > highest_backbone_level:
            self.extra_convs = paddle.nn.LayerList()
            if self.use_c5:
                in_c = in_channels[0]
                fan = in_c * 3 * 3
            else:
                in_c = self.num_chan
                fan = in_c * 3 * 3
            for i in range(highest_backbone_level + 1, self.max_level + 1):
                conv = Conv2dUnit(in_c, self.num_chan, 3, stride=2, bias_attr=True, bn=0, gn=0, af=0, bias_lr=2.0,
                                  weight_init=Xavier(fan_out=fan), bias_init=Constant(0.0),
                                  act=None, freeze_norm=self.freeze_norm, norm_decay=self.norm_decay, name='fpn_%d' % (i, ))
                self.extra_convs.append(conv)


    def forward(self, body_feats):
        '''
        一个示例
        :param body_feats:  [s8, s16, s32]
        :return:
                                     bs32
                                      |
                                     卷积
                                      |
                             bs16   [fs32]
                              |       |
                            卷积    上采样
                              |       |
                          lateral   topdown
                               \    /
                                相加
                                  |
                        bs8     [fs16]
                         |        |
                        卷积    上采样
                         |        |
                      lateral   topdown
                            \    /
                             相加
                               |
                             [fs8]

                fpn_inner_output = [fs32, fs16, fs8]
        然后  fs32, fs16, fs8  分别再接一个卷积得到 p5, p4, p3 ；
        p5 接一个卷积得到 p6， p6 接一个卷积得到 p7。
        '''
        spatial_scale = copy.deepcopy(self.spatial_scale)
        num_backbone_stages = self.num_backbone_stages   # 进入FPN的张量个数
        body_feats = body_feats[-1:-num_backbone_stages - 1:-1]   # 倒序。 [s32, s16, s8, ...]
        fpn_inner_output = [None] * num_backbone_stages
        fpn_inner_output[0] = self.fpn_inner_convs[0](body_feats[0])
        for i in range(1, num_backbone_stages):
            body_input = body_feats[i]
            top_output = fpn_inner_output[i - 1]
            fpn_inner_single = self._add_topdown_lateral(i, body_input, top_output)
            fpn_inner_output[i] = fpn_inner_single
        fpn_output = [None] * num_backbone_stages
        for i in range(num_backbone_stages):
            fpn_output[i] = self.fpn_convs[i](fpn_inner_output[i])

        # 生成其它尺度的特征图时如果用的是池化层
        if not self.has_extra_convs and self.max_level - self.min_level == len(
                spatial_scale):
            body_top_extension = fluid.layers.pool2d(
                fpn_output[0],
                1,
                'max',
                pool_stride=2)
            fpn_output.insert(0, body_top_extension)
            spatial_scale.insert(0, spatial_scale[0] * 0.5)

        # 生成其它尺度的特征图时如果用的是卷积层
        highest_backbone_level = self.min_level + len(spatial_scale) - 1
        if self.has_extra_convs and self.max_level > highest_backbone_level:
            if self.use_c5:
                fpn_blob = body_feats[0]
            else:
                fpn_blob = fpn_output[0]
            for i in range(highest_backbone_level + 1, self.max_level + 1):
                fpn_blob_in = fpn_blob
                if i > highest_backbone_level + 1:
                    fpn_blob_in = fluid.layers.relu(fpn_blob)
                fpn_blob = self.extra_convs[i - highest_backbone_level - 1](fpn_blob_in)
                fpn_output.insert(0, fpn_blob)
                spatial_scale.insert(0, spatial_scale[0] * 0.5)

        if self.reverse_out:
            fpn_output = fpn_output[::-1]  # 倒序。
        return fpn_output, spatial_scale


    def _add_topdown_lateral(self, i, body_input, upper_output):
        lateral = self.fpn_inner_convs[i](body_input)
        if body_input.shape[2] == -1 and body_input.shape[3] == -1:
            topdown = fluid.layers.resize_nearest(upper_output, scale=2.)
        else:
            topdown = fluid.layers.resize_nearest(
                upper_output,
                out_shape=[body_input.shape[2], body_input.shape[3]])

        return lateral + topdown



