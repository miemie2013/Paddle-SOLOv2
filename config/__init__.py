#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-11-21 09:13:23
#   Description : paddle2.0_solov2
#
# ================================================================
from . import get_model
from . import solov2_r50_fpn_8gpu_3x
from . import solov2_light_448_r50_fpn_8gpu_3x
from . import solov2_light_r50_vd_fpn_dcn_512_3x

from .get_model import *
from .solov2_r50_fpn_8gpu_3x import *
from .solov2_light_448_r50_fpn_8gpu_3x import *
from .solov2_light_r50_vd_fpn_dcn_512_3x import *
