#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================
from config import *
from model.solo import *
import torch
import paddle
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



cfg = SOLOv2_r50_fpn_8gpu_3x_Config()



def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('SOLOv2_R50_3x.pth')
print('============================================================')

backbone_dic = {}
fpn_dic = {}
head_dic = {}
mask_feat_head_dic = {}
others = {}
for key, value in state_dict['state_dict'].items():
    if 'tracked' in key:
        continue
    if 'backbone' in key:
        backbone_dic[key] = value.data.numpy()
    elif 'neck' in key:
        fpn_dic[key] = value.data.numpy()
    elif 'bbox_head' in key:
        head_dic[key] = value.data.numpy()
    elif 'mask_feat_head' in key:
        mask_feat_head_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

print()




# 创建模型
Backbone = select_backbone(cfg.backbone_type)
backbone = Backbone(**cfg.backbone)
FPN = select_fpn(cfg.fpn_type)
fpn = FPN(**cfg.fpn)
MaskFeatHead = select_head(cfg.mask_feat_head_type)
mask_feat_head = MaskFeatHead(**cfg.mask_feat_head)
Head = select_head(cfg.head_type)
head = Head(solo_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)
model = SOLOv2(backbone, fpn, mask_feat_head, head)

model.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式。
param_state_dict = model.state_dict()

print('\nCopying...')



def copy(name, w):
    value2 = paddle.to_tensor(w, place=place)
    value = param_state_dict[name]
    value = value * 0 + value2
    param_state_dict[name] = value

def copy_conv_bn(conv_unit_name, w, scale, offset, m, v):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.bn.weight', scale)
    copy(conv_unit_name + '.bn.bias', offset)
    copy(conv_unit_name + '.bn._mean', m)
    copy(conv_unit_name + '.bn._variance', v)


def copy_conv(conv_name, w, b):
    copy(conv_name + '.weight', w)
    copy(conv_name + '.bias', b)


def copy_conv_gn(conv_unit_name, w, scale, offset):
    copy(conv_unit_name + '.conv.weight', w)
    copy(conv_unit_name + '.gn.weight', scale)
    copy(conv_unit_name + '.gn.bias', offset)



# Resnet50Vb
w = backbone_dic['backbone.conv1.weight']
scale = backbone_dic['backbone.bn1.weight']
offset = backbone_dic['backbone.bn1.bias']
m = backbone_dic['backbone.bn1.running_mean']
v = backbone_dic['backbone.bn1.running_var']
copy_conv_bn('backbone.stage1_conv1_1', w, scale, offset, m, v)



nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)

        conv_name1 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name1 + '.conv1.weight']
        scale = backbone_dic[conv_name1 + '.bn1.weight']
        offset = backbone_dic[conv_name1 + '.bn1.bias']
        m = backbone_dic[conv_name1 + '.bn1.running_mean']
        v = backbone_dic[conv_name1 + '.bn1.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv1' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        conv_name2 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name2 + '.conv2.weight']
        scale = backbone_dic[conv_name2 + '.bn2.weight']
        offset = backbone_dic[conv_name2 + '.bn2.bias']
        m = backbone_dic[conv_name2 + '.bn2.running_mean']
        v = backbone_dic[conv_name2 + '.bn2.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv2' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        conv_name3 = 'backbone.layer%d.%d' % ((nid+1), kk)
        w = backbone_dic[conv_name3 + '.conv3.weight']
        scale = backbone_dic[conv_name3 + '.bn3.weight']
        offset = backbone_dic[conv_name3 + '.bn3.bias']
        m = backbone_dic[conv_name3 + '.bn3.running_mean']
        v = backbone_dic[conv_name3 + '.bn3.running_var']
        conv_unit_name = 'backbone.stage%d_%d.conv3' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            shortcut_name = 'backbone.layer%d.%d.downsample' % ((nid + 1), kk)
            w = backbone_dic[shortcut_name + '.0.weight']
            scale = backbone_dic[shortcut_name + '.1.weight']
            offset = backbone_dic[shortcut_name + '.1.bias']
            m = backbone_dic[shortcut_name + '.1.running_mean']
            v = backbone_dic[shortcut_name + '.1.running_var']
            conv_unit_name = 'backbone.stage%d_%d.conv4' % (2+nid, kk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


# fpn
w = fpn_dic['neck.lateral_convs.3.conv.weight']
b = fpn_dic['neck.lateral_convs.3.conv.bias']
copy_conv('fpn.fpn_inner_convs.0.conv', w, b)

w = fpn_dic['neck.lateral_convs.2.conv.weight']
b = fpn_dic['neck.lateral_convs.2.conv.bias']
copy_conv('fpn.fpn_inner_convs.1.conv', w, b)

w = fpn_dic['neck.lateral_convs.1.conv.weight']
b = fpn_dic['neck.lateral_convs.1.conv.bias']
copy_conv('fpn.fpn_inner_convs.2.conv', w, b)

w = fpn_dic['neck.lateral_convs.0.conv.weight']
b = fpn_dic['neck.lateral_convs.0.conv.bias']
copy_conv('fpn.fpn_inner_convs.3.conv', w, b)

w = fpn_dic['neck.fpn_convs.3.conv.weight']
b = fpn_dic['neck.fpn_convs.3.conv.bias']
copy_conv('fpn.fpn_convs.0.conv', w, b)

w = fpn_dic['neck.fpn_convs.2.conv.weight']
b = fpn_dic['neck.fpn_convs.2.conv.bias']
copy_conv('fpn.fpn_convs.1.conv', w, b)

w = fpn_dic['neck.fpn_convs.1.conv.weight']
b = fpn_dic['neck.fpn_convs.1.conv.bias']
copy_conv('fpn.fpn_convs.2.conv', w, b)

w = fpn_dic['neck.fpn_convs.0.conv.weight']
b = fpn_dic['neck.fpn_convs.0.conv.bias']
copy_conv('fpn.fpn_convs.3.conv', w, b)


# head
num_convs = 4
for lvl in range(0, num_convs):
    # conv + gn
    w = head_dic['bbox_head.kernel_convs.%d.conv.weight'%lvl]
    scale = head_dic['bbox_head.kernel_convs.%d.gn.weight'%lvl]
    offset = head_dic['bbox_head.kernel_convs.%d.gn.bias'%lvl]
    copy_conv_gn('head.krn_convs.%d' % (lvl, ), w, scale, offset)

    # conv + gn
    w = head_dic['bbox_head.cate_convs.%d.conv.weight'%lvl]
    scale = head_dic['bbox_head.cate_convs.%d.gn.weight'%lvl]
    offset = head_dic['bbox_head.cate_convs.%d.gn.bias'%lvl]
    copy_conv_gn('head.cls_convs.%d' % (lvl, ), w, scale, offset)

# 类别分支最后的conv
w = head_dic['bbox_head.solo_cate.weight']
b = head_dic['bbox_head.solo_cate.bias']
copy_conv('head.cls_convs.%d.conv' % (num_convs,), w, b)

# 卷积核分支最后的conv
w = head_dic['bbox_head.solo_kernel.weight']
b = head_dic['bbox_head.solo_kernel.bias']
copy_conv('head.krn_convs.%d.conv' % (num_convs,), w, b)



# mask_feat_head
start_level = 0
end_level = 3
for i in range(start_level, end_level + 1):
    if i == 0:
        w = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0.conv.weight' % (i,)]
        scale = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0.gn.weight' % (i,)]
        offset = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0.gn.bias' % (i,)]
        copy_conv_gn('mask_feat_head.convs_all_levels.%d.0' % (i,), w, scale, offset)
        continue

    for j in range(i):
        w = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d.conv.weight' % (i, j)]
        scale = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d.gn.weight' % (i, j)]
        offset = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d.gn.bias' % (i, j)]
        copy_conv_gn('mask_feat_head.convs_all_levels.%d.%d' % (i, j * 2), w, scale, offset)


w = mask_feat_head_dic['mask_feat_head.conv_pred.0.conv.weight']
scale = mask_feat_head_dic['mask_feat_head.conv_pred.0.gn.weight']
offset = mask_feat_head_dic['mask_feat_head.conv_pred.0.gn.bias']
copy_conv_gn('mask_feat_head.conv_pred', w, scale, offset)




save_name = 'dygraph_solov2_r50_fpn_8gpu_3x.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







