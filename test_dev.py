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
from tools.cocotools import get_classes, catid2clsid, clsid2catid
import os
import argparse
import textwrap
import paddle
import json

from tools.cocotools import eval
from model.decode_np import Decode
from model.solo import *
from tools.argparser import ArgParser
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = ArgParser()
    use_gpu = parser.get_use_gpu()
    cfg = parser.get_cfg()
    print(paddle.__version__)
    paddle.disable_static()   # 开启动态图
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

    # 读取的模型
    model_path = cfg.eval_cfg['model_path']

    # 是否给图片画框。
    draw_image = cfg.eval_cfg['draw_image']
    draw_thresh = cfg.eval_cfg['draw_thresh']

    # 验证时的批大小
    eval_batch_size = cfg.eval_cfg['eval_batch_size']

    # 打印，确认一下使用的配置
    print('\n=============== config message ===============')
    print('config file: %s' % str(type(cfg)))
    print('model_path: %s' % model_path)
    print('target_size: %d' % cfg.eval_cfg['target_size'])
    print('use_gpu: %s' % str(use_gpu))
    print()

    # test集图片的相对路径
    test_pre_path = cfg.test_pre_path
    anno_file = cfg.test_path
    with open(anno_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            images = dataset['images']

    # 种类id
    _catid2clsid = {}
    _clsid2catid = {}
    _clsid2cname = {}
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:
        dataset_text = ''
        for line in f2:
            line = line.strip()
            dataset_text += line
        eval_dataset = json.loads(dataset_text)
        categories = eval_dataset['categories']
        for clsid, cate_dic in enumerate(categories):
            catid = cate_dic['id']
            cname = cate_dic['name']
            _catid2clsid[catid] = clsid
            _clsid2catid[clsid] = catid
            _clsid2cname[clsid] = cname
    class_names = []
    num_classes = len(_clsid2cname.keys())
    for clsid in range(num_classes):
        class_names.append(_clsid2cname[clsid])


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

    param_state_dict = paddle.load(model_path)
    model.set_state_dict(param_state_dict)
    model.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式。
    head.set_dropblock(is_test=True)

    _decode = Decode(model, class_names, place, cfg, for_test=False)
    eval(_decode, images, test_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image, draw_thresh, type='test_dev')

