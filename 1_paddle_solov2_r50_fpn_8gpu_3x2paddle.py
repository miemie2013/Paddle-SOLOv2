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
import paddle
import os

print(paddle.__version__)
paddle.disable_static()
# 开启动态图



use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()



cfg = SOLOv2_r50_fpn_8gpu_3x_Config()
# 该模型是COCO数据集上训练好的，所以强制改类别数为80
cfg.num_classes = 80
cfg.head['num_classes'] = cfg.num_classes + 1
model_path = 'solov2_r50_fpn_3x.pdparams'



def load_weights(path):
    state_dict = fluid.io.load_program_state(path)
    return state_dict

state_dict = load_weights(model_path)
print('============================================================')

backbone_dic = {}
fpn_dic = {}
head_dic = {}
mask_feat_head_dic = {}
others = {}
for key, value in state_dict.items():
    # if 'tracked' in key:
    #     continue
    if 'branch' in key:
        backbone_dic[key] = value
    elif 'fpn' in key:
        fpn_dic[key] = value
    elif 'bbox_head' in key:
        head_dic[key] = value
    elif 'mask_feat_head' in key:
        mask_feat_head_dic[key] = value
    else:
        others[key] = value

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
w = state_dict['conv1_weights']
scale = state_dict['bn_conv1_scale']
offset = state_dict['bn_conv1_offset']
m = state_dict['bn_conv1_mean']
v = state_dict['bn_conv1_variance']
copy_conv_bn('backbone.stage1_conv1_1', w, scale, offset, m, v)



nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)

        conv_name1 = block_name + "_branch2a"
        bn_name1 = 'bn' + conv_name1[3:]
        w = backbone_dic[conv_name1 + '_weights']
        scale = backbone_dic[bn_name1 + '_scale']
        offset = backbone_dic[bn_name1 + '_offset']
        m = backbone_dic[bn_name1 + '_mean']
        v = backbone_dic[bn_name1 + '_variance']
        conv_unit_name = 'backbone.stage%d_%d.conv1' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        conv_name2 = block_name + "_branch2b"
        bn_name2 = 'bn' + conv_name2[3:]
        w = backbone_dic[conv_name2 + '_weights']
        scale = backbone_dic[bn_name2 + '_scale']
        offset = backbone_dic[bn_name2 + '_offset']
        m = backbone_dic[bn_name2 + '_mean']
        v = backbone_dic[bn_name2 + '_variance']
        conv_unit_name = 'backbone.stage%d_%d.conv2' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


        conv_name3 = block_name + "_branch2c"
        bn_name3 = 'bn' + conv_name3[3:]
        w = backbone_dic[conv_name3 + '_weights']
        scale = backbone_dic[bn_name3 + '_scale']
        offset = backbone_dic[bn_name3 + '_offset']
        m = backbone_dic[bn_name3 + '_mean']
        v = backbone_dic[bn_name3 + '_variance']
        conv_unit_name = 'backbone.stage%d_%d.conv3' % (2+nid, kk)
        copy_conv_bn(conv_unit_name, w, scale, offset, m, v)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            shortcut_name = block_name + "_branch1"
            shortcut_bn_name = 'bn' + shortcut_name[3:]
            w = backbone_dic[shortcut_name + '_weights']
            scale = backbone_dic[shortcut_bn_name + '_scale']
            offset = backbone_dic[shortcut_bn_name + '_offset']
            m = backbone_dic[shortcut_bn_name + '_mean']
            v = backbone_dic[shortcut_bn_name + '_variance']
            conv_unit_name = 'backbone.stage%d_%d.conv4' % (2+nid, kk)
            copy_conv_bn(conv_unit_name, w, scale, offset, m, v)


# fpn
w = fpn_dic['fpn_inner_res5_sum_w']
b = fpn_dic['fpn_inner_res5_sum_b']
copy_conv('fpn.fpn_inner_convs.0.conv', w, b)

w = fpn_dic['fpn_inner_res4_sum_lateral_w']
b = fpn_dic['fpn_inner_res4_sum_lateral_b']
copy_conv('fpn.fpn_inner_convs.1.conv', w, b)

w = fpn_dic['fpn_inner_res3_sum_lateral_w']
b = fpn_dic['fpn_inner_res3_sum_lateral_b']
copy_conv('fpn.fpn_inner_convs.2.conv', w, b)

w = fpn_dic['fpn_inner_res2_sum_lateral_w']
b = fpn_dic['fpn_inner_res2_sum_lateral_b']
copy_conv('fpn.fpn_inner_convs.3.conv', w, b)

w = fpn_dic['fpn_res5_sum_w']
b = fpn_dic['fpn_res5_sum_b']
copy_conv('fpn.fpn_convs.0.conv', w, b)

w = fpn_dic['fpn_res4_sum_w']
b = fpn_dic['fpn_res4_sum_b']
copy_conv('fpn.fpn_convs.1.conv', w, b)

w = fpn_dic['fpn_res3_sum_w']
b = fpn_dic['fpn_res3_sum_b']
copy_conv('fpn.fpn_convs.2.conv', w, b)

w = fpn_dic['fpn_res2_sum_w']
b = fpn_dic['fpn_res2_sum_b']
copy_conv('fpn.fpn_convs.3.conv', w, b)


# head
num_convs = 4
for lvl in range(0, num_convs):
    # conv + gn
    w = head_dic['bbox_head.kernel_convs.%d_weights'%lvl]
    scale = head_dic['bbox_head.kernel_convs.%d.gn_scale'%lvl]
    offset = head_dic['bbox_head.kernel_convs.%d.gn_offset'%lvl]
    copy_conv_gn('head.krn_convs.%d' % (lvl, ), w, scale, offset)

    # conv + gn
    w = head_dic['bbox_head.cate_convs.%d_weights'%lvl]
    scale = head_dic['bbox_head.cate_convs.%d.gn_scale'%lvl]
    offset = head_dic['bbox_head.cate_convs.%d.gn_offset'%lvl]
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
        w = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0_weights' % (i,)]
        scale = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0.gn_scale' % (i,)]
        offset = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv0.gn_offset' % (i,)]
        copy_conv_gn('mask_feat_head.convs_all_levels.%d.0' % (i,), w, scale, offset)
        continue

    for j in range(i):
        w = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d_weights' % (i, j)]
        scale = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d.gn_scale' % (i, j)]
        offset = mask_feat_head_dic['mask_feat_head.convs_all_levels.%d.conv%d.gn_offset' % (i, j)]
        copy_conv_gn('mask_feat_head.convs_all_levels.%d.%d' % (i, j * 2), w, scale, offset)


w = mask_feat_head_dic['mask_feat_head.conv_pred.0_weights']
scale = mask_feat_head_dic['mask_feat_head.conv_pred.0.gn_scale']
offset = mask_feat_head_dic['mask_feat_head.conv_pred.0.gn_offset']
copy_conv_gn('mask_feat_head.conv_pred', w, scale, offset)




save_name = 'dygraph_solov2_r50_fpn_8gpu_3x.pdparams'
paddle.save(param_state_dict, save_name)
print('\nDone.')







