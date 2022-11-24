
"""
The heads definition of DATE.

Author:
    Yiqun Chen
"""

from copy import deepcopy
from typing import Dict, List, Sequence, Tuple
from warnings import warn

from mmcv.cnn import Scale
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_head
from mmdet.models.dense_heads.fcos_head import AnchorFreeHead
from torch import Tensor, nn

from date.models.modules.identity import Identity
from date.models.predictors.base_predictor import BaseHeadPredictor
from date.models.predictors.one2one_predictor import OneToOneHeadPredictor

INF = 1E8


@HEADS.register_module()
class DATEHead(AnchorFreeHead):
    """
    NMS free head.

    Loss will be calculated by predictors.
    """
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            feat_channels: int = 256,
            stacked_convs: int = 4,
            share_stacked_convs: bool = True,
            strides: Sequence[int] = (8, 16, 32, 64, 128),
            deformable: bool = False,
            predict_from: int = 0,
            predictors: List[Dict] = [
                dict(type='OneToOneHeadPredictor',
                     loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=2.0),
                     loss_box=dict(type='L1Loss', loss_weight=5.0),
                     loss_iou=dict(type='GIoULoss', loss_weight=2.0))],
            conv_cfg: Dict = dict(
                type='Conv2d',
                deformable=dict(
                    type='ModulatedDeformableConv2d',
                    cls_convs=list(),
                    reg_convs=list()),
                ),
            norm_cfg: Dict = dict(
                type='GN', num_groups=32, requires_grad=True),
            train_cfg: Dict = None,
            test_cfg: Dict = None,
            init_cfg: Dict = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01),
            **kwargs):
        self.deformable = deformable
        if predict_from >= len(predictors) \
                or predict_from < -len(predictors):
            raise ValueError(
                f'NMS Free head can only predict from [{-len(predictors)} '
                f'to {len(predictors)-1}], but got {predict_from}.')
        self.predict_from = predict_from
        self.share_stacked_convs = share_stacked_convs
        self.num_predictors = len(predictors)
        assert self.num_predictors > 0
        strides = tuple(strides)
        super().__init__(
            num_classes,
            in_channels,
            feat_channels,
            stacked_convs,
            strides=strides,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self._format_predictors(predictors)
        predictors = [build_head(p) for p in predictors]
        self.predictors = nn.ModuleList(predictors)

    def init_weights(self):
        super().init_weights()
        for p in self.predictors:
            assert isinstance(p, BaseHeadPredictor)
            if hasattr(p, 'init_weights'):
                p.init_weights()

    def forward(
        self, feats: Tuple[Tensor, ...]
    ) -> Tuple[Tuple[Tuple[List[Tensor], ...], ...]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns: A list of predictions of each predictor with following format:
            tuple:
                cls_probs (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                reg_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                ...
        """
        # cls_feat, reg_feat = super().forward(feats)
        cls_feats, reg_feats = self._forward_stacked_convs(feats)
        if self.num_predictors == 1 or self.share_stacked_convs:
            preds = [tuple(p(cls_feats, reg_feats, self.scales, self.strides))
                     for p in self.predictors]
        else:
            preds = [tuple(p(c, r, self.scales, self.strides))
                     for p, c, r in zip(self.predictors, cls_feats, reg_feats)]
        return (tuple(preds), )

    def loss(self,
             preds: Tuple[Tuple[List[Tensor], ...]],
             gt_bboxes: List[Tensor],
             gt_labels: List[Tensor],
             img_metas: List[Dict],
             gt_bboxes_ignore=None) -> Dict[str, Tensor]:
        """
        Loss for predictions of multiple predictors.

        Args:
            preds: The predictions of multiple predictors. Each element
                of list is contains outputs of corresponding predictor
                (cls_probs: Tensor, reg_preds: Tensor, ...).
            gt_bboxes: Bounding boxes of each image with shape (G, 4).
            gt_labels: Labels of each image with shape (G, ).
            img_metas: Meta information of each image.
        """
        assert len(preds) == len(self.predictors), \
               f'The number of predictions is not equal to' \
               f'the number of predictors.'
        loss_dict = dict()
        for idx, (pred, predictor) in enumerate(zip(preds, self.predictors)):
            num_loss = len(loss_dict)
            assert isinstance(predictor, BaseHeadPredictor)
            loss_pred: Dict = predictor.loss(
                *pred, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
            for key, val in loss_pred.items():
                loss_dict[f'loss.{idx}.{key}'] = val
            assert (num_loss + len(loss_pred)) == len(loss_dict), \
                   f'Some losses are overwritten.'
        return loss_dict

    def get_bboxes(
            self,
            preds: Tuple[Tuple[List[Tensor], ...]],
            score_factors: List[Tensor] = None,
            img_metas: List[Dict] = None,
            cfg: Dict = None,
            rescale: bool = False,
            with_nms: bool = False,
            **kwargs):
        predictor = self.predictors[self.predict_from]
        assert isinstance(predictor, BaseHeadPredictor)
        warn(f'You are using {predictor.__class__} to perform inference.')
        return predictor.get_bboxes(
            *preds[self.predict_from],
            score_factors=score_factors,
            img_metas=img_metas,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            **kwargs)

    def _forward_stacked_convs(self, feats: Tuple[Tensor, ...]):
        if self.share_stacked_convs:
            return self._forward_shared_stacked_convs(feats)
        else:
            return self._forward_exclusive_stacked_convs(feats)

    def _forward_shared_stacked_convs(
            self, feats: Tuple[Tensor, ...]) \
            -> Tuple[List[Tensor], List[Tensor]]:
        # return super().forward(feats)
        cls_feats = self._forward_stacked_convs_single_branch(
            self.cls_convs, feats)
        reg_feats = self._forward_stacked_convs_single_branch(
            self.reg_convs, feats)
        return cls_feats, reg_feats

    def _forward_exclusive_stacked_convs(self, feats: Tuple[Tensor, ...]) \
            -> Tuple[Tuple[List[Tensor], List[Tensor]], ...]:
        cls_feats = multi_apply(
            self._forward_stacked_convs_single_branch,
            self.cls_convs,
            [feats for _ in range(len(self.cls_convs))])
        cls_feats = [
            [cls_feats[idx_f][idx_p] for idx_f in range(len(feats))]
            for idx_p in range(self.num_predictors)]

        reg_feats = multi_apply(
            self._forward_stacked_convs_single_branch,
            self.reg_convs,
            [feats for _ in range(len(self.reg_convs))])
        reg_feats = [
            [reg_feats[idx_f][idx_p] for idx_f in range(len(feats))]
            for idx_p in range(self.num_predictors)]

        return (cls_feats, reg_feats)

    def _forward_stacked_convs_single_branch(
            self, stacked_convs: nn.Module, feats: Tuple[Tensor, ...]) \
            -> Tuple[List[Tensor], List[Tensor]]:
        return multi_apply(
            self._forward_stacked_convs_single_branch_single_level,
            [stacked_convs for _ in range(len(feats))],
            feats)[0]

    def _forward_stacked_convs_single_branch_single_level(
            self, stacked_convs: nn.Module, feat: Tensor) -> Tensor:
        for conv in stacked_convs:
            feat = conv(feat)
        return (feat, )

    def get_targets(self, *args, **kwargs):
        raise NotImplementedError('not implemented.')

    def _init_predictor(self):
        # Prediction is delegated to self.predictors.
        self.conv_cls = Identity()
        self.conv_reg = Identity()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        # Such logic make code be compatible with old parameters.
        if self.num_predictors == 1 or self.share_stacked_convs:
            self.cls_convs = self._build_stacked_convs('cls_convs')
        else:
            self.cls_convs = nn.ModuleList(
                [self._build_stacked_convs('cls_convs')
                 for _ in range(self.num_predictors)])

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        # Such logic make code be compatible with old parameters.
        if self.num_predictors == 1 or self.share_stacked_convs:
            self.reg_convs = self._build_stacked_convs('reg_convs')
        else:
            self.reg_convs = nn.ModuleList(
                [self._build_stacked_convs('reg_convs')
                 for _ in range(self.num_predictors)])

    def _build_stacked_convs(self, module_name: str) -> nn.ModuleList:
        if self.stacked_convs == 0:
            return Identity()
        stacked_convs = nn.ModuleList()
        assert isinstance(self.conv_cfg, dict)
        conv_cfg: Dict = self.conv_cfg.pop('deformable', dict())
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.deformable and i in conv_cfg[module_name]:
                conv_cfg = dict(
                    type=conv_cfg.get(
                        'type', 'ModulatedDeformableConv2d'))
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False,
                    bias=self.conv_bias))
        return stacked_convs

    def _format_predictors(self, predictors: List[Dict]):
        for p in predictors:
            p['num_classes'] = self.num_classes
            p['in_channels'] = self.feat_channels
            p['strides'] = self.strides
            for k in p.keys():
                if isinstance(p[k], dict) and 'strides' in p[k].keys():
                    p[k]['strides'] = self.strides
