
"""
Definition of the one-to-many assignment branch (RetinaNet-style).

Author:
    Yiqun Chen
"""

from typing import Dict, Union, List
from warnings import warn

from torch import Tensor
from mmcv.cnn import Scale
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.retina_head import RetinaHead
from date.models.predictors.base_predictor import BaseHeadPredictor


class _RetinaHeadPredictor(RetinaHead):
    def _init_layers(self):
        pass


@HEADS.register_module()
class RetinaHeadPredictor(BaseHeadPredictor):
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            deformable: bool = False,
            loss_cls: Dict = dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_iou: Dict = None,
            loss_box: Dict = dict(type='L1Loss', loss_weight=2.0),
            train_cfg: Dict = dict(),
            test_cfg: Dict = dict(),
            init_cfg: Union[Dict, List[Dict]] = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal',
                    name='conv_cls',
                    std=0.01,
                    bias_prob=0.01)),
            **kwargs):
        retina_head = _RetinaHeadPredictor(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_box,
            stacked_convs=0,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        self.num_base_priors = retina_head.num_base_priors
        self.cls_out_channels = retina_head.cls_out_channels
        super().__init__(
            num_classes,
            in_channels,
            deformable,
            loss_cls,
            loss_iou,
            loss_box,
            train_cfg,
            test_cfg,
            init_cfg,
            **kwargs)
        self.retina_head = retina_head
        self.loss_cls = self.retina_head.loss_cls
        self.loss_box = self.retina_head.loss_bbox

    def forward_single(self,
                       cls_feat: Tensor,
                       reg_feat: Tensor,
                       scale: Scale,
                       *args) -> Tensor:
        cls_pred = self.conv_cls(cls_feat)
        reg_pred = self.conv_reg(reg_feat)
        reg_pred = scale(reg_pred)
        return cls_pred, reg_pred

    def loss(self,
             cls_preds: List[Tensor],
             reg_preds: List[Tensor],
             gt_bboxes: List[Tensor],
             gt_labels: List[Tensor],
             img_metas: List[Dict],
             gt_bboxes_ignore: List[Tensor] = None) -> Dict[str, Tensor]:
        loss_dict = self.retina_head.loss(
            cls_preds,
            reg_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore)
        return dict(loss_cls=loss_dict['loss_cls'],
                    loss_box=loss_dict['loss_bbox'])

    def _build_cls_layer(self):
        layer = self._get_layer_type()
        conv_cls_out_channels = \
            self.num_base_priors \
            * self.cls_out_channels
        self.conv_cls = layer(
            self.in_channels,
            conv_cls_out_channels,
            3,
            padding=1)

    def _build_reg_layer(self):
        layer = self._get_layer_type()
        self.conv_reg = layer(
            self.in_channels,
            self.num_base_priors * 4,
            3,
            padding=1)

    def get_bboxes(self, *args, **kwargs):
        score_factors = kwargs.pop('score_factors', None)
        if score_factors is not None:
            raise ValueError(
                f'RetinaHeadPredictor does not recognize'
                f'specified score_factors.')
        with_nms = True if self.test_cfg.get('nms', True) else False
        kwargs.pop('with_nms')  # Specified by cfg.
        if not with_nms:
            warn('You are performing inference without NMS.')
        return self.retina_head.get_bboxes(*args, with_nms=with_nms, **kwargs)
