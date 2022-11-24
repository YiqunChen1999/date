
"""Definition of Dynamic Deformable Convolution.

Author:
    Yiqun Chen
"""

import torch
from mmcv.cnn.bricks.registry import CONV_LAYERS
from mmcv.ops.modulated_deform_conv import (ModulatedDeformConv2dPack,
                                            modulated_deform_conv2d)


# DeformConv2d from torchvision.ops requires extra mask as input during
# `forward` to enable the version 2, thus here the ModulatedDeformConv2d
# is preferred.
CONV_LAYERS.register_module(
    name='ModulatedDeformConv2dPack', module=ModulatedDeformConv2dPack)


@CONV_LAYERS.register_module()
class ModulatedDeformableConv2d(ModulatedDeformConv2dPack):
    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        super().init_weights()

    def init_weights(self) -> None:
        None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)  # (y, x, y, x, ...)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
