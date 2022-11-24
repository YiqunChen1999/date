
"""This file defines the identity modules.

Author:
    Yiqun Chen
"""

from warnings import warn

from mmcv.runner import BaseModule
from mmdet.models.builder import MODELS


@MODELS.register_module()
class Identity(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        warn(f'Configs {kwargs} for Identity module will not be used.')

    def forward(self, x, *args, **kwargs):
        return x
