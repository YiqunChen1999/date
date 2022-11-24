
"""
Base predictor.

Author:
    Yiqun Chen
"""

from typing import List, Tuple, Union, Dict

from torch import nn, Tensor
from mmcv.cnn import Scale, initialize
from mmcv.runner import force_fp32
from mmcv.utils.logging import print_log, logger_initialized
from mmdet.core import multi_apply
from mmdet.models.builder import build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
from date.models.modules.conv import ModulatedDeformableConv2d
from date.utils.utils import sort_and_keep_topk_results


class BaseHeadPredictor(BaseDenseHead, BBoxTestMixin):
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            deformable: bool = False,
            loss_cls: Dict = None,
            loss_iou: Dict = None,
            loss_box: Dict = None,
            train_cfg: Dict = dict(),
            test_cfg: Dict = dict(),
            init_cfg: Union[Dict, List[Dict]] = [
                dict(
                    type='Normal',
                    layer='Conv2d',
                    std=0.01,
                    override=dict(
                        type='Normal',
                        name='conv_cls',
                        std=0.01,
                        bias_prob=0.01)),
                dict(
                    type='Normal',
                    layer='ModulatedDeformableConv2d',
                    std=0.01,
                    override=dict(
                        type='Normal',
                        name='conv_cls',
                        std=0.01,
                        bias_prob=0.01))],
            **kwargs):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.deformable = deformable
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.loss_cls: nn.Module = self._build_loss(loss_cls)
        self.loss_iou: nn.Module = self._build_loss(loss_iou)
        self.loss_box: nn.Module = self._build_loss(loss_box)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._build_model()
        self._positive_indices = list()
        self._negative_indices = list()
        self._assigned_indices = list()

    def init_weights(self):
        module_name = self.__class__.__name__
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmcv'
        print_log(
            f'initialize {module_name} with init_cfg {self.init_cfg}',
            logger=logger_name)
        initialize(self, self.init_cfg)

    def forward(
            self,
            cls_feat: Union[Tensor, List[Tensor]],
            reg_feat: Union[Tensor, List[Tensor]],
            scale: Union[Scale, List[Scale]],
            stride: Union[int, List[int]],
    ) -> Tuple[List[Tensor], ...]:
        if isinstance(cls_feat, Tensor) and isinstance(reg_feat, Tensor):
            return self.forward_single(cls_feat,
                                       reg_feat,
                                       scale,
                                       stride)
        else:
            return multi_apply(
                self.forward_single, cls_feat, reg_feat, scale, stride)

    def forward_single(self,
                       cls_feat: Tensor,
                       reg_feat: Tensor,
                       scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor]:
        if not isinstance(cls_feat, Tensor) \
                or not isinstance(reg_feat, Tensor):
            raise TypeError(
                f'BaseHeadPredictor processes single level feature map, '
                f'which should be type Tensor, but got '
                f'type of cls_feat: {type(cls_feat)}, '
                f'type of reg_feat: {type(reg_feat)}.')
        cls_pred = self.conv_cls(cls_feat)
        reg_pred = self.conv_reg(reg_feat)
        reg_pred: Tensor = scale(reg_pred)
        reg_pred = reg_pred.clamp(min=0)
        reg_pred = reg_pred * stride
        assert reg_pred.max() < 1e5, f'{reg_pred.max()}'
        return cls_pred, reg_pred

    def _build_loss(self, loss: Dict):
        return build_loss(loss) if loss else None

    def _build_model(self):
        self._build_cls_layer()
        self._build_reg_layer()

    def _build_reg_layer(self):
        layer = self._get_layer_type()
        self.conv_reg = layer(self.in_channels, 4, 3, padding=1)

    def _build_cls_layer(self):
        layer = self._get_layer_type()
        self.conv_cls = layer(self.in_channels, self.num_classes, 3, padding=1)

    def _get_layer_type(self):
        if self.deformable:
            return ModulatedDeformableConv2d
        else:
            return nn.Conv2d

    @force_fp32(apply_to=('cls_probs_list', 'reg_preds_list'))
    def get_bboxes(
            self,
            cls_probs: List[Tensor],
            reg_preds: List[Tensor],
            score_factors: List[Tensor] = None,
            img_metas: List[Dict] = None,
            cfg: Dict = None,
            rescale: bool = False,
            with_nms: bool = False,
            **kwargs):
        nms = self.test_cfg.get('nms', None)
        with_nms = nms is not None and nms is not False
        results = super(BaseHeadPredictor, self).get_bboxes(
            cls_probs,
            reg_preds,
            score_factors,
            img_metas,
            cfg,
            rescale,
            with_nms,
            **kwargs)
        assert len(results) == len(img_metas)
        if not with_nms:
            max_per_img = self.test_cfg.get('max_per_img', 100)
            return sort_and_keep_topk_results(results, max_per_img)
        return results

    def _clear_assigned_cache(self):
        if len(self._positive_indices):
            self._positive_indices.clear()
        if len(self._negative_indices):
            self._negative_indices.clear()
        if len(self._assigned_indices):
            self._assigned_indices.clear()
