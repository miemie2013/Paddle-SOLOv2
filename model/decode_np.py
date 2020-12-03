#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================
import random
import colorsys
import cv2
import threading
import os
import paddle
import paddle.nn.functional as F
import paddle.fluid.layers as L
import numpy as np

from tools.transform import *


class Decode(object):
    def __init__(self, model, all_classes, place, cfg, for_test=True):
        self.all_classes = all_classes
        self.num_classes = len(self.all_classes)
        self.model = model
        self.place = place

        # 图片预处理
        self.context = cfg.context
        # sample_transforms
        self.to_rgb = cfg.decodeImage['to_rgb']
        target_size = cfg.eval_cfg['target_size']
        if for_test:
            target_size = cfg.test_cfg['target_size']
        self.resizeImage = ResizeImage(target_size=target_size, resize_box=False, interp=cfg.resizeImage['interp'],
                                       max_size=cfg.resizeImage['max_size'], use_cv2=cfg.resizeImage['use_cv2'])
        self.normalizeImage = NormalizeImage(**cfg.normalizeImage)
        self.permute = Permute(**cfg.permute)
        # batch_transforms
        self.padBatch = PadBatch(use_padded_im_info=False, pad_to_stride=cfg.padBatch['pad_to_stride'])


    # 处理一张图片
    def detect_image(self, image, pimage, ori_shape, resize_shape, draw_image, draw_thresh=0.0):
        pred = self.predict(pimage, ori_shape, resize_shape)
        if pred['scores'][0] < 0:
            boxes = np.array([])
            masks = np.array([])
            scores = np.array([])
            classes = np.array([])
        else:
            masks = pred['masks']
            scores = pred['scores']
            classes = pred['classes'].astype(np.int32)
            # 获取boxes
            boxes = []
            for ms in masks:
                sum_1 = np.sum(ms, axis=0)
                x = np.where(sum_1 > 0.5)[0]
                sum_2 = np.sum(ms, axis=1)
                y = np.where(sum_2 > 0.5)[0]
                if len(x) == 0:  # 掩码全是0的话（即没有一个像素是前景）
                    x0, x1, y0, y1 = 0, 1, 0, 1
                else:
                    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
                boxes.append([x0, y0, x1, y1])
            boxes = np.array(boxes).astype(np.float32)
        if len(scores) > 0 and draw_image:
            pos = np.where(scores >= draw_thresh)
            boxes2 = boxes[pos]         # [M, 4]
            scores2 = scores[pos]       # [M, ]
            classes2 = classes[pos]     # [M, ]
            masks2 = masks[pos]         # [M, h, w]
            self.draw(image, boxes2, scores2, classes2, masks2)
        return image, boxes, scores, classes

    # 处理一批图片
    def detect_batch(self, batch_img, batch_pimage, batch_ori_shape, batch_resize_shape, draw_image, draw_thresh=0.0):
        batch_size = len(batch_img)
        result_image, result_boxes, result_scores, result_classes, result_masks = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size

        pred = self.predict(batch_pimage, batch_ori_shape, batch_resize_shape)
        if pred['scores'][0] < 0:
            boxes = np.array([])
            masks = np.array([])
            scores = np.array([])
            classes = np.array([])
        else:
            masks = pred['masks']
            scores = pred['scores']
            classes = pred['classes'].astype(np.int32)
            # 获取boxes
            boxes = []
            for ms in masks:
                sum_1 = np.sum(ms, axis=0)
                x = np.where(sum_1 > 0.5)[0]
                sum_2 = np.sum(ms, axis=1)
                y = np.where(sum_2 > 0.5)[0]
                if len(x) == 0:  # 掩码全是0的话（即没有一个像素是前景）
                    x0, x1, y0, y1 = 0, 1, 0, 1
                else:
                    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
                boxes.append([x0, y0, x1, y1])
            boxes = np.array(boxes).astype(np.float32)
        if len(scores) > 0 and draw_image:
            pos = np.where(scores >= draw_thresh)
            boxes2 = boxes[pos]         # [M, 4]
            scores2 = scores[pos]       # [M, ]
            classes2 = classes[pos]     # [M, ]
            masks2 = masks[pos]         # [M, h, w]
            self.draw(batch_img[0], boxes2, scores2, classes2, masks2)

        i = 0
        result_image[i] = batch_img[i]
        result_boxes[i] = boxes
        result_scores[i] = scores
        result_classes[i] = classes
        result_masks[i] = masks
        return result_image, result_boxes, result_scores, result_classes, result_masks

    def draw(self, image, boxes, scores, classes, masks, mask_alpha=0.45):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl, ms in zip(boxes, scores, classes, masks):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))

            # 随机颜色
            bbox_color = random.choice(colors)
            # 同一类别固定颜色
            # bbox_color = colors[cl]

            # 在这里上掩码颜色。咩咩深度优化的画掩码代码。
            color = np.array(bbox_color)
            color = np.reshape(color, (1, 1, 3))
            target_ms = ms[top:bottom, left:right]
            target_ms = np.expand_dims(target_ms, axis=2)
            target_ms = np.tile(target_ms, (1, 1, 3))
            target_region = image[top:bottom, left:right, :]
            target_region = target_ms * (target_region * (1 - mask_alpha) + color * mask_alpha) + (
                        1 - target_ms) * target_region
            image[top:bottom, left:right, :] = target_region


            # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            bbox_thick = 1
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    def process_image(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.resizeImage(sample, context)
        sample = self.normalizeImage(sample, context)
        sample = self.permute(sample, context)

        # batch_transforms
        samples = self.padBatch([sample], context)
        sample = samples[0]

        pimage = np.expand_dims(sample['image'], axis=0)
        ori_shape = np.array([[sample['h'], sample['w']]]).astype(np.int32)
        resize_shape = np.array([[sample['im_info'][0], sample['im_info'][1]]]).astype(np.int32)
        return pimage, ori_shape, resize_shape

    def predict(self, image, ori_shape, resize_shape):
        image = paddle.to_tensor(image, place=self.place)
        ori_shape = paddle.to_tensor(ori_shape, place=self.place)
        resize_shape = paddle.to_tensor(resize_shape, place=self.place)
        preds = self.model(image, ori_shape, resize_shape)
        numpy_preds = {}
        for key in preds.keys():
            value = preds[key]
            value = value.numpy()
            numpy_preds[key] = value
        return numpy_preds



