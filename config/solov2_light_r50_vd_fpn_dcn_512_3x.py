#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================



class SOLOv2_light_r50_vd_fpn_dcn_512_3x_Config(object):
    def __init__(self):
        # 自定义数据集
        # self.train_path = 'annotation_json/voc2012_train.json'
        # self.val_path = 'annotation_json/voc2012_val.json'
        # self.classes_path = 'data/voc_classes.txt'
        # self.train_pre_path = '../data/data4379/pascalvoc/VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        # self.val_pre_path = '../data/data4379/pascalvoc/VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径
        # self.num_classes = 20                              # 数据集类别数

        # AIStudio下的COCO数据集
        self.train_path = '../data/data7122/annotations/instances_train2017.json'
        self.val_path = '../data/data7122/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = '../data/data7122/train2017/'  # 训练集图片相对路径
        self.val_pre_path = '../data/data7122/val2017/'      # 验证集图片相对路径
        self.test_path = '../data/data7122/annotations/image_info_test-dev2017.json'      # test集
        self.test_pre_path = '../data/data7122/test2017/'    # test集图片相对路径
        self.num_classes = 80                                # 数据集类别数

        # Windows下的COCO数据集
        # self.train_path = '../COCO/annotations/instances_train2017.json'
        # self.val_path = '../COCO/annotations/instances_val2017.json'
        # self.classes_path = 'data/coco_classes.txt'
        # self.train_pre_path = '../COCO/train2017/'  # 训练集图片相对路径
        # self.val_pre_path = '../COCO/val2017/'      # 验证集图片相对路径
        # self.test_path = '../COCO/annotations/image_info_test-dev2017.json'      # test集
        # self.test_pre_path = '../COCO/test2017/'    # test集图片相对路径
        # self.num_classes = 80                       # 数据集类别数


        # ========= 一些设置 =========
        self.train_cfg = dict(
            batch_size=12,
            num_threads=5,   # 读数据的线程数
            max_batch=2,     # 最大读多少个批
            # model_path='dygraph_solov2_light_r50_vd_fpn_dcn_512_3x.pdparams',
            model_path='dygraph_r50vd_ssld.pdparams',
            # model_path='./weights/1000.pdparams',
            save_iter=1000,   # 每隔几步保存一次模型
            eval_iter=30000,   # 每隔几步计算一次eval集的mAP
            max_iters=270000,   # 训练多少步
            mixup_epoch=10,     # 前几轮进行mixup
            cutmix_epoch=-1,    # 前几轮进行cutmix
        )
        self.learningRate = dict(
            base_lr=0.01,
            PiecewiseDecay=dict(
                gamma=0.1,
                milestones=[180000, 240000],
            ),
            LinearWarmup=dict(
                start_factor=0.,
                steps=1000,
            ),
        )
        self.optimizerBuilder = dict(
            optimizer=dict(
                momentum=0.9,
                type='Momentum',
            ),
            regularizer=dict(
                factor=0.0001,
                type='L2',
            ),
        )


        # 验证。用于train.py、eval.py、test_dev.py
        self.eval_cfg = dict(
            model_path='dygraph_solov2_light_r50_vd_fpn_dcn_512_3x.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=512,
            draw_image=False,    # 是否画出验证集图片
            draw_thresh=0.15,    # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
            eval_batch_size=1,   # 验证时的批大小。
        )

        # 测试。用于demo.py
        self.test_cfg = dict(
            model_path='dygraph_solov2_light_r50_vd_fpn_dcn_512_3x.pdparams',
            # model_path='./weights/1000.pdparams',
            target_size=512,
            # target_size=320,
            draw_image=True,
            draw_thresh=0.15,   # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
        )


        # ============= 模型相关 =============
        self.use_ema = True
        # self.use_ema = False
        self.ema_decay = 0.9998
        self.backbone_type = 'Resnet50Vd'
        self.backbone = dict(
            norm_type='sync_bn',
            feature_maps=[2, 3, 4, 5],
            dcn_v2_stages=[3, 4, 5],
            downsample_in3x3=True,   # 注意这个细节，是在3x3卷积层下采样的。
            freeze_at=0,
            freeze_norm=False,
            norm_decay=0.,
            lr_mult_list=[0.05, 0.05, 0.1, 0.15],
        )
        self.fpn_type = 'FPN'
        self.fpn = dict(
            in_channels=[2048, 1024, 512, 256],
            num_chan=256,
            min_level=2,
            max_level=6,
            spatial_scale=[0.03125, 0.0625, 0.125, 0.25],
            has_extra_convs=False,
            use_c5=False,
            reverse_out=True,
        )
        self.mask_feat_head_type = 'MaskFeatHead'
        self.mask_feat_head = dict(
            in_channels=256,
            out_channels=128,
            norm_type='gn',
            start_level=0,
            end_level=3,
            num_classes=128,
        )
        self.head_type = 'SOLOv2Head'
        self.head = dict(
            num_classes=self.num_classes + 1,
            in_channels=256,
            norm_type='gn',
            num_convs=3,
            seg_feat_channels=256,
            strides=[8, 8, 16, 32, 32],
            sigma=0.2,
            kernel_out_channels=128,
            num_grids=[40, 36, 24, 16, 12],
            dcn_v2_stages=[2],
            drop_block=True,
        )
        self.solo_loss_type = 'SOLOv2Loss'
        self.solo_loss = dict(
            ins_loss_weight=3.0,
            focal_loss_gamma=2.0,
            focal_loss_alpha=0.25,
        )
        self.nms_cfg = dict(
            score_thr=0.1,
            update_thr=0.05,
            mask_thr=0.5,
            nms_pre=500,
            max_per_img=100,
            kernel="gaussian",
            sigma=2.,
        )


        # ============= 预处理相关 =============
        self.context = {'fields': ['image', 'im_id', 'gt_segm']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=False,
            with_cutmix=False,
        )
        # Poly2Mask
        self.poly2Mask = dict(
        )
        # ColorDistort
        self.colorDistort = dict(
        )
        # RandomCrop
        self.randomCrop = dict(
            is_mask_crop=True,
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=[352, 384, 416, 448, 480, 512],
            max_size=852,
            interp=1,
            use_cv2=True,
            resize_box=True,
        )
        # RandomFlipImage
        self.randomFlipImage = dict(
            prob=0.5,
        )
        # NormalizeImage
        self.normalizeImage = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            is_scale=False,
            is_channel_first=False,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )
        # PadBatch
        self.padBatch = dict(
            pad_to_stride=32,
        )
        # Gt2Solov2Target
        self.gt2Solov2Target = dict(
            num_grids=[40, 36, 24, 16, 12],
            scale_ranges=[[1, 64], [32, 128], [64, 256], [128, 512], [256, 2048]],
            coord_sigma=0.2,
        )

        # 预处理顺序。增加一些数据增强时这里也要加上，否则train.py中相当于没加！
        self.sample_transforms_seq = []
        self.sample_transforms_seq.append('decodeImage')
        self.sample_transforms_seq.append('poly2Mask')
        self.sample_transforms_seq.append('colorDistort')
        self.sample_transforms_seq.append('randomCrop')
        self.sample_transforms_seq.append('resizeImage')
        self.sample_transforms_seq.append('randomFlipImage')
        self.sample_transforms_seq.append('normalizeImage')
        self.sample_transforms_seq.append('permute')
        self.batch_transforms_seq = []
        self.batch_transforms_seq.append('padBatch')
        self.batch_transforms_seq.append('gt2Solov2Target')


