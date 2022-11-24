
"""
Definition of one-to-many assignment branch (FCOS-style).

Author:
    Yiqun Chen
"""

from typing import Callable, Sequence, Union, List, Dict
from warnings import warn

from torch import Tensor
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.fcos_head import FCOSHead
from date.models.predictors.base_predictor import BaseHeadPredictor


INF = 1E8


class _FCOSHeadPredictor(FCOSHead):
    def _init_layers(self):
        pass


@HEADS.register_module()
class FCOSHeadPredictor(BaseHeadPredictor):
    """
    An auxiliary head with FCOS.
    """
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            strides: Sequence[int] = [8, 16, 32, 64, 128],
            regress_ranges: Sequence[Sequence[int]] = (
                (-1, 64), (64, 128), (128, 256), (256, 512), (512, 1E8)),
            center_sampling: bool = False,
            center_sample_radius: bool = 1.5,
            centerness_on_reg: bool = False,
            norm_on_bbox: bool = False,
            infer_without_center: bool = False,
            loss_cls: Dict = None,
            loss_box: Dict = None,
            loss_ctr: Dict = None,
            deformable: bool = False,
            init_cfg: Union[Dict, List[Dict]] = [
                dict(type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                dict(type='Normal',
                     layer='ModulatedDeformableConv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01))],
            **kwargs):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            deformable=deformable,
            loss_cls=loss_cls,
            loss_box=loss_box,
            init_cfg=init_cfg,
            **kwargs)
        if infer_without_center:
            loss_ctr['loss_weight'] = 0.0
        self.fcos_head = _FCOSHeadPredictor(
            num_classes=num_classes,
            in_channels=in_channels,
            regress_ranges=regress_ranges,
            strides=strides,
            center_sampling=center_sampling,
            center_sample_radius=center_sample_radius,
            norm_on_bbox=norm_on_bbox,
            centerness_on_reg=centerness_on_reg,
            loss_cls=loss_cls,
            loss_bbox=loss_box,
            loss_centerness=loss_ctr,
            init_cfg=None,
            **kwargs)
        self.infer_without_center = infer_without_center
        self.loss_cls = self.fcos_head.loss_cls
        self.loss_ctr = self.fcos_head.loss_centerness
        self.loss_box = self.fcos_head.loss_bbox

    def forward_single(
            self,
            cls_feat: Tensor,
            reg_feat: Tensor,
            scale: Callable[[Tensor], Tensor],
            stride: int) -> Tensor:
        if not isinstance(cls_feat, Tensor) \
                or not isinstance(reg_feat, Tensor):
            raise TypeError(
                f'AuxiliaryGaussianHead processes single level feature map, '
                f'which should be type Tensor, but got '
                f'type of cls_feat: {type(cls_feat)}, '
                f'type of reg_feat: {type(reg_feat)}.')
        # Follow FCOSHead.
        if self.fcos_head.centerness_on_reg:
            ctr_pred: Tensor = self.conv_ctr(reg_feat)
        else:
            ctr_pred: Tensor = self.conv_ctr(cls_feat)
        ctr_pred = ctr_pred[:, 0, ...].unsqueeze(1)
        assert len(ctr_pred.shape) == 4 and ctr_pred.shape[1] == 1, \
               f'Unexpected ctr_pred.shape: {ctr_pred.shape}'

        cls_pred = self.conv_cls(cls_feat)
        reg_pred = self.conv_reg(reg_feat)
        reg_pred = scale(reg_pred).float()
        if self.fcos_head.norm_on_bbox:
            # reg_pred needed for gradient computation has been modified
            # by F.relu(reg_pred) when run with PyTorch 1.10. So replace
            # F.relu(reg_pred) with reg_pred.clamp(min=0)
            reg_pred = reg_pred.clamp(min=0)
            if not self.training:
                reg_pred *= stride
        else:
            reg_pred = reg_pred.exp()
        return cls_pred, reg_pred, ctr_pred

    def loss(self,
             cls_preds: List[Tensor],
             box_preds: List[Tensor],
             ctr_preds: List[Tensor],
             gt_bboxes: List[Tensor],
             gt_labels: List[Tensor],
             img_metas: List[Dict],
             gt_bboxes_ignore: List[Tensor] = None) -> Dict[str, Tensor]:
        loss_dict = self.fcos_head.loss(
            cls_preds,
            box_preds,
            ctr_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore)
        return dict(
            loss_fcos_cls=loss_dict['loss_cls'],
            loss_fcos_iou=loss_dict['loss_bbox'],
            loss_fcos_ctr=loss_dict['loss_centerness'])

    def get_bboxes(self, *args, **kwargs):
        score_factors = kwargs.pop('score_factors', None)
        if score_factors is not None:
            raise ValueError(
                f'FCOSHeadPredictor does not recognize'
                f'specified score_factors.')
        with_nms = True if self.test_cfg.get('nms', True) else False
        kwargs.pop('with_nms')  # Specified by cfg.
        if self.infer_without_center:
            warn(f'Center-ness branch is discarded during inference.')
            args = args[: 2] + (None, )
        return self.fcos_head.get_bboxes(*args, with_nms=with_nms, **kwargs)

    def _build_model(self):
        self._build_cls_layer()
        self._build_reg_layer()
        self._build_ctr_layer()

    def _init_layers(self):
        pass

    def _build_ctr_layer(self):
        layer = self._get_layer_type()
        self.conv_ctr = layer(self.in_channels, 1, 3, padding=1)
