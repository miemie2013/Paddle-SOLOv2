#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import argparse
import textwrap
from config import *


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Script', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu. True or False')
        parser.add_argument('-c', '--config', type=int, default=2,
                            choices=[0, 1, 2],
                            help=textwrap.dedent('''\
                            select one of these config files:
                            0 -- solov2_r50_fpn_8gpu_3x.py
                            1 -- solov2_light_448_r50_fpn_8gpu_3x.py
                            2 -- solov2_light_r50_vd_fpn_dcn_512_3x.py'''))
        self.args = parser.parse_args()
        self.config_file = self.args.config
        self.use_gpu = self.args.use_gpu

    def get_use_gpu(self):
        return self.use_gpu

    def get_cfg(self):
        config_file = self.config_file
        cfg = None
        if config_file == 0:
            cfg = SOLOv2_r50_fpn_8gpu_3x_Config()
        elif config_file == 1:
            cfg = SOLOv2_light_448_r50_fpn_8gpu_3x_Config()
        elif config_file == 2:
            cfg = SOLOv2_light_r50_vd_fpn_dcn_512_3x_Config()
        return cfg


