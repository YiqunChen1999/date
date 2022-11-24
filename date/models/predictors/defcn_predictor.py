
"""
Modified by Yiqun Chen

See: https://github.com/Megvii-BaseDetection/DeFCN
"""

from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models.builder import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from torch import Tensor, nn

from date.models.predictors.one2one_predictor import OneToOneHeadPredictor
from date.utils.grid import get_points_in_boxes_mask


@HEADS.register_module()
class DeFCNPredictor(OneToOneHeadPredictor):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 poto_alpha: float = 0.8,
                 poto_aux_topk: int = 9,
                 strides: Sequence[int] = (8, 16, 32, 64, 128),
                 bbox_coder: Dict = dict(type='DistancePointBBoxCoder'),
                 deformable: bool = False,
                 loss_cls: Dict = None,
                 loss_iou: Dict = None,
                 loss_box: Dict = None,
                 loss_aux: Dict = None,
                 loss_box_format: str = 'xywh',
                 prior_generator: Dict = dict(
                    type='MlvlPointGenerator',
                    strides=(8, 16, 32, 64, 128)),
                 assigner: Dict = dict(
                    type='POTOAssigner',
                    cls_cost=dict(
                        type='FocalLossCost', weight=2.0),
                    reg_cost=dict(
                        type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    iou_cost=dict(
                        type='IoUCost', iou_mode='giou', weight=2.0)),
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
        super().__init__(num_classes=num_classes,
                         in_channels=in_channels,
                         strides=strides,
                         bbox_coder=bbox_coder,
                         deformable=deformable,
                         loss_cls=loss_cls,
                         loss_iou=loss_iou,
                         loss_box=loss_box,
                         loss_box_format=loss_box_format,
                         prior_generator=prior_generator,
                         assigner=assigner,
                         init_cfg=init_cfg,
                         **kwargs)
        # Following the original setting.
        self.loss_aux = self._build_loss(loss_aux)
        self.poto_alpha = poto_alpha
        self.poto_aux_topk = poto_aux_topk
        self.max3d = MaxFiltering(in_channels=in_channels,
                                  kernel_size=3,
                                  tau=2)
        self.filter = nn.Conv2d(in_channels=in_channels,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self,
                cls_feat: Union[Tensor, List[Tensor]],
                reg_feat: Union[Tensor, List[Tensor]],
                scale: Union[Scale, List[Scale]],
                stride: Union[int, List[int]]) -> Union[Tensor, List[Tensor]]:
        cls_preds, reg_preds = \
            super().forward(cls_feat, reg_feat, scale, stride)
        filtering: List[Tensor] = self.max3d(reg_feat)
        filtering: List[Tensor] = [self.filter(x) for x in filtering]
        return cls_preds, reg_preds, filtering

    @force_fp32(apply_to=('cls_probs_list', 'reg_preds_list'))
    def loss(self,
             cls_probs_list: List[Tensor],
             reg_preds_list: List[Tensor],
             filtering_list: List[Tensor],
             gt_bboxes_list: List[Tensor],
             gt_labels_list: List[Tensor],
             img_metas_list: List[Dict],
             gt_bboxes_ignore: List[Tensor] = None) -> Dict[str, Tensor]:
        """Compute loss of the head with One-to-One matching.

        Args:
            cls_probs: Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            reg_preds: Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            gt_bboxes: Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels: class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # # To not break our framework.
        cls_probs_list = self._filter_boxes(cls_probs_list, filtering_list)
        loss_cls, loss_iou, loss_box = self._calc_one_to_one_loss(
            cls_probs_list,
            reg_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas_list,
            gt_bboxes_ignore)
        loss_dict = dict()
        loss_dict['loss_o2o_cls'] = loss_cls
        loss_dict['loss_o2o_box'] = loss_box
        loss_dict['loss_o2o_iou'] = loss_iou
        if self.loss_aux is not None:
            loss_aux = self._calc_aux_one_to_many_loss(
                cls_probs_list=cls_probs_list,
                reg_preds_list=reg_preds_list,
                gt_bboxes_list=gt_bboxes_list,
                gt_labels_list=gt_labels_list,
                img_metas=img_metas_list,
                gt_bboxes_ignore=gt_bboxes_ignore)
            loss_dict['loss_o2m_aux'] = loss_aux
        return loss_dict

    def _calc_aux_one_to_many_loss(
            self,
            cls_probs_list: List[Tensor],
            reg_preds_list: List[Tensor],
            gt_bboxes_list: List[Tensor],
            gt_labels_list: List[Tensor],
            img_metas: List[Dict],
            gt_bboxes_ignore: List[Tensor] = None) -> Dict[str, Tensor]:
        batch_size = len(img_metas)
        featmap_sizes = self._calc_featmap_sizes(cls_probs_list)
        anchor_points, num_points_per_level = self._generate_anchor_points(
            featmap_sizes, cls_probs_list[0].dtype, cls_probs_list[0].device)
        cls_probs, box_preds = self._calc_cls_box_prds(
            anchor_points, cls_probs_list, reg_preds_list)

        (
            labels_targets_list,
            labels_weights_list,
            pos_mask_list,
            neg_mask_list) = self._get_aux_one_to_many_targets(
                cls_probs=cls_probs,
                box_preds=box_preds,
                gt_bboxes_list=gt_bboxes_list,
                gt_labels_list=gt_labels_list,
                img_metas=img_metas,
                points=[anchor_points for _ in range(batch_size)],
                gt_bboxes_ignore=gt_bboxes_ignore,
                num_points_per_level=num_points_per_level)

        labels_targets = torch.cat(labels_targets_list, 0)
        labels_weights = torch.cat(labels_weights_list, 0)

        # Calculate average factor for loss.
        cls_avg_factor = self._calc_aux_cls_avg_weight(pos_mask_list)

        loss_aux = self.loss_aux(cls_probs.flatten(0, 1),
                                 labels_targets,
                                 labels_weights,
                                 avg_factor=cls_avg_factor)
        return loss_aux

    def _calc_aux_cls_avg_weight(self, pos_mask_list: List[Tensor]) -> Tensor:
        device = pos_mask_list[0].device
        num_total_pos = sum([mask.sum() for mask in pos_mask_list])
        num_total_pos = torch.as_tensor(
            [num_total_pos], dtype=torch.float, device=device)
        num_total_pos = reduce_mean(num_total_pos)
        cls_avg_factor = num_total_pos
        cls_avg_factor = max(cls_avg_factor, 1)
        return cls_avg_factor

    @torch.no_grad()
    def _get_aux_one_to_many_targets(
            self,
            cls_probs: Tensor,
            box_preds: Tensor,
            gt_bboxes_list: List[Tensor],
            gt_labels_list: List[Tensor],
            img_metas: List[Dict],
            points: List[Tensor] = None,
            gt_bboxes_ignore: List[Tensor] = None,
            num_points_per_level: List[Tensor] = None):
        assert gt_bboxes_ignore is None
        num_imgs = len(gt_bboxes_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_imgs)]
        (
            labels_targets_list,
            labels_weights_list,
            pos_mask_list,
            neg_mask_list) = multi_apply(
                self._get_aux_one_to_many_targets_single_image,
                cls_probs,
                box_preds,
                gt_bboxes_list,
                gt_labels_list,
                img_metas,
                points,
                gt_bboxes_ignore_list,
                num_points_per_level=num_points_per_level)
        return (labels_targets_list,
                labels_weights_list,
                pos_mask_list,
                neg_mask_list)

    def _get_aux_one_to_many_targets_single_image(
            self,
            cls_probs: Tensor,
            box_preds: Tensor,
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            img_meta: Dict,
            points: Tensor = None,
            gt_bboxes_ignore: Tensor = None,
            num_points_per_level: List[Tensor] = None):
        """
        Calculate the targets using methods described in DeFCN paper.

        Args:
            cls_probs: Flattened classification scores for one image.
            reg_probs: Flattened regression bounding boxes for one image.
            gt_bboxes:
            gt_labels: Min valid value is 0.
            img_meta:
            points: Flattened points coordinates of each location
                of all level features.
            ...
        """
        # Both inputs should be in format (x, y, x, y).
        cls_quality = \
            cls_probs[:, gt_labels].sigmoid() ** (1 - self.poto_alpha)
        iou_mat = bbox_overlaps(box_preds, gt_bboxes)
        box_quality = iou_mat ** self.poto_alpha
        assert cls_quality.shape == box_quality.shape, \
               f'Shape mismatch: {cls_quality.shape} == {box_quality.shape}.'
        quality = cls_quality * box_quality  # (N, G)

        # Select candidates per level.
        idx_start_cur_lvl = 0
        candidates_indices = []
        for num_points_cur_lvl in num_points_per_level:
            idx_end_cur_lvl = idx_start_cur_lvl + num_points_cur_lvl
            _, topk_ids = torch.topk(
                quality[idx_start_cur_lvl: idx_end_cur_lvl],  # Ni x G
                self.poto_aux_topk,
                dim=0)
            candidates_indices.append(topk_ids)
            idx_start_cur_lvl = idx_end_cur_lvl
        # (num_level*topk, G)
        candidates_indices = torch.cat(candidates_indices, dim=0)
        points_in_boxes_mask = get_points_in_boxes_mask(points, gt_bboxes)
        # (num_level*topk, G)
        candidates_qualities = torch.gather(quality,
                                            dim=0,
                                            index=candidates_indices)
        # (1, G)
        quality_threshold = \
            candidates_qualities.mean(dim=0, keepdim=True) \
            + candidates_qualities.std(dim=0, keepdim=True)
        positive_mask = torch.zeros_like(points_in_boxes_mask)
        positive_mask = positive_mask.scatter_(0,
                                               candidates_indices,
                                               True)
        positive_mask &= (quality >= quality_threshold)

        quality[~points_in_boxes_mask] = -1
        quality[~positive_mask] = -1

        # For a location, it may still have more than one ground-truth,
        # select the one with max quality.
        quality4points, matched_gt_ids = quality.max(dim=1)

        num_pos = (quality4points != -1).sum().item()
        num_gts = len(gt_labels)

        num_bboxes = box_preds.shape[0]
        labels_targets = gt_labels.new_full((num_bboxes, ),
                                            self.num_classes,
                                            dtype=torch.long)
        labels_weights = gt_labels.new_ones(num_bboxes)

        pos_mask = quality4points != -1  # defcn treat -1 as negative.
        neg_mask = quality4points == -1  # defcn treat -1 as negative.
        labels_targets[pos_mask] = gt_labels[matched_gt_ids[pos_mask]]

        return labels_targets, labels_weights, pos_mask, neg_mask

    def get_bboxes(self,
                   cls_probs_list: List[Tensor],
                   reg_preds_list: List[Tensor],
                   filtering_list: List[Tensor],
                   *args,
                   **kwargs):
        score_factors = kwargs.pop('score_factors', None)
        if score_factors is not None:
            raise ValueError(
                f'FCOSHeadPredictor does not recognize'
                f'specified score_factors.')
        with_nms = True if self.test_cfg.get('nms', True) else False
        kwargs.pop('with_nms')  # Specified by cfg.
        cls_probs_list = self._filter_boxes(cls_probs_list, filtering_list)
        return super().get_bboxes(cls_probs_list,
                                  reg_preds_list,
                                  *args,
                                  with_nms=with_nms,
                                  **kwargs)

    def _filter_boxes(self,
                      cls_probs_list: List[Tensor],
                      filtering_list: List[Tensor]) -> List[Tensor]:
        cls_probs_list = tuple([inverse_sigmoid(c.sigmoid()*f.sigmoid())
                                for c, f in zip(cls_probs_list,
                                                filtering_list)])
        return cls_probs_list


class MaxFiltering(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3, tau: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm = nn.GroupNorm(32, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size),
            padding=(tau // 2, kernel_size // 2, kernel_size // 2),
            stride=1
        )
        self.margin = tau // 2

    def forward(self, inputs: Tuple[Tensor]):
        features = []
        for _, x in enumerate(inputs):
            features.append(self.conv(x))

        outputs = []
        for lvl, x in enumerate(features):
            assert isinstance(x, Tensor)
            # Assign lambda function to a var seems not a good idea.
            # func = lambda f: F.interpolate(f,
            #                                size=x.shape[2:],
            #                                mode="bilinear")
            func = upsample
            size = x.shape[2:]
            feature_3d = []
            for k in range(max(0, lvl - self.margin),
                           min(len(features), lvl + self.margin + 1)):
                feature_3d.append(
                    func(features[k], size) if k != lvl else features[k])

            feature_3d = torch.stack(feature_3d, dim=2)
            max_pool = self.max_pool(feature_3d)[:, :, min(lvl, self.margin)]
            output = max_pool + inputs[lvl]
            outputs.append(self.nonlinear(self.norm(output)))
        return outputs


def upsample(feat: Tensor, size: Tuple[int, int]):
    return F.interpolate(feat, size, mode='bilinear')
