
"""
Definition of the one-to-one assignment branch.

Author:
    Yiqun Chen
"""

from typing import Dict, List, Sequence, Tuple, Union
from warnings import warn

import torch
from mmcv.runner import force_fp32
from mmdet.core import (bbox_xyxy_to_cxcywh, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.core.bbox.assigners import (AssignResult, BaseAssigner,
                                       HungarianAssigner)
from mmdet.core.bbox.coder import BaseBBoxCoder
from mmdet.core.bbox.samplers import BaseSampler, SamplingResult
from mmdet.models.builder import HEADS
from torch import Tensor

from date.models.modules.assigner import POTOAssigner
from date.models.predictors.base_predictor import BaseHeadPredictor


@HEADS.register_module()
class OneToOneHeadPredictor(BaseHeadPredictor):
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            strides: Sequence[int] = (8, 16, 32, 64, 128),
            bbox_coder: Dict = dict(type='DistancePointBBoxCoder'),
            deformable: bool = False,
            loss_cls: Dict = None,
            loss_iou: Dict = None,
            loss_box: Dict = None,
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
        assert tuple(strides) == tuple(prior_generator['strides']), \
               f'strides not equal: {strides} = {prior_generator["strides"]}'
        self.prior_generator = build_prior_generator(prior_generator)
        self.bbox_coder: BaseBBoxCoder = build_bbox_coder(bbox_coder)
        self.loss_box_format = loss_box_format
        self.assigner: BaseAssigner = build_assigner(assigner)
        self.sampler: BaseSampler = build_sampler(
            dict(type='PseudoSampler', context=self))
        super().__init__(
            num_classes,
            in_channels,
            deformable,
            loss_cls=loss_cls,
            loss_iou=loss_iou,
            loss_box=loss_box,
            init_cfg=init_cfg,
            **kwargs)

    @force_fp32(apply_to=('cls_probs_list', 'reg_preds_list'))
    def loss(self,
             cls_probs_list: List[Tensor],
             reg_preds_list: List[Tensor],
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
            aux_cls_probs: Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            aux_reg_preds: Box energies / deltas for each scale
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
        return loss_dict

    def _calc_one_to_one_loss(
            self,
            cls_probs_list: List[Tensor],
            reg_preds_list: List[Tensor],
            gt_bboxes_list: List[Tensor],
            gt_labels_list: List[Tensor],
            img_metas: List[Dict],
            gt_bboxes_ignore: List[Tensor] = None) -> Dict[str, Tensor]:
        """Compute loss of the head with One-to-One matching.

        Args:
            cls_probs_list: Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            reg_preds_list: Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            ctr_preds_list: centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
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
        assert len(cls_probs_list) == len(reg_preds_list)

        batch_size = len(img_metas)
        featmap_sizes = self._calc_featmap_sizes(cls_probs_list)
        anchor_points, num_points_per_level = self._generate_anchor_points(
            featmap_sizes, cls_probs_list[0].dtype, cls_probs_list[0].device)
        cls_probs, box_preds = self._calc_cls_box_prds(
            anchor_points, cls_probs_list, reg_preds_list)

        (  # Get targets.
            labels_targets_list,
            labels_weights_list,
            bboxes_targets_list,
            bboxes_weights_list,
            pos_inds_list,
            neg_inds_list,
            factor_list,
            pos_assigned_gt_inds_list) = self.get_targets(
                cls_probs=cls_probs,
                box_preds=box_preds,
                gt_bboxes_list=gt_bboxes_list,
                gt_labels_list=gt_labels_list,
                img_metas=img_metas,
                points=[anchor_points for _ in range(batch_size)],
                gt_bboxes_ignore=gt_bboxes_ignore,
                num_points_per_level=num_points_per_level)

        # Cat batches.
        labels_targets = torch.cat(labels_targets_list, 0)
        labels_weights = torch.cat(labels_weights_list, 0)
        bboxes_targets = torch.cat(bboxes_targets_list, 0)
        bboxes_weights = torch.cat(bboxes_weights_list, 0)

        # Calculate average factor for loss.
        cls_avg_factor = self._calc_cls_avg_weight(
            pos_inds_list, neg_inds_list)
        box_avg_factor = self._calc_box_avg_weight(bboxes_targets)

        # Get normalized bboxes for L1Loss.
        normed_box_prds, normed_box_trgs = self._get_normed_prds_trgs(
            box_preds, bboxes_targets_list, factor_list)

        loss_cls: Tensor = self.loss_cls(
            cls_probs.flatten(0, 1),
            labels_targets,
            labels_weights,
            avg_factor=cls_avg_factor)
        loss_iou = self.loss_iou(
            box_preds.flatten(0, 1),
            bboxes_targets,
            bboxes_weights,
            avg_factor=box_avg_factor)
        loss_box = self.loss_box(
            normed_box_prds.flatten(0, 1),
            normed_box_trgs,
            bboxes_weights,
            avg_factor=box_avg_factor)

        return loss_cls, loss_iou, loss_box

    def _calc_featmap_sizes(
            self, preds_list: List[Tensor]) -> List[torch.Size]:
        return [featmap.size()[-2:] for featmap in preds_list]

    def _calc_cls_box_prds(
            self,
            anchors: Tensor,
            cls_probs_list: List[Tensor],
            reg_preds_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
        batch_size = cls_probs_list[0].shape[0]
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)

        reg_preds = self._cat_and_bchw_to_bnc(reg_preds_list)  # (B, N, 4)
        box_preds: Tensor = self.bbox_coder.decode(anchors, reg_preds)

        cls_probs = self._cat_and_bchw_to_bnc(cls_probs_list)  # (B, N, C)

        return cls_probs, box_preds

    def _cat_and_bchw_to_bnc(self, data_list: List[Tensor]) -> Tensor:
        data = torch.cat([
            c.flatten(2, 3) for c in data_list], dim=2)
        data = data.permute(0, 2, 1)  # (B, N, C)
        return data

    def _generate_anchor_points(
            self,
            featmap_sizes: List[int],
            dtype: torch.dtype,
            device: torch.device) -> Tuple[Tensor, List[int]]:
        all_level_points: List[Tensor] = self.prior_generator.grid_priors(
            featmap_sizes, dtype=dtype, device=device)
        flattened_points = torch.cat(all_level_points, dim=0)  # (N, 2)
        num_points_per_level = [center.size(0) for center in all_level_points]
        return flattened_points, num_points_per_level

    def _get_normed_prds_trgs(
            self,
            box_prds: Tensor,
            box_trgs_list: List[Tensor],
            factor_list: List[Tensor]):
        """
        Get normalized predictions and targets.

        Args:
            box_prds: Unnormalized bboxes predictions in format (x, y, x, y).
            box_trgs_list: Unnormalized targets list for each scales
                in format (x, y, x, y).
            factor_list: Factor list for each image.
        """
        factor = torch.cat(factor_list, 0).unsqueeze(1)  # (B, 1, 4)
        # DETR regress the relative (normalized) boxes in format (x, y, w, h).
        normed_box_trgs = [
            trg / f for trg, f in zip(box_trgs_list, factor)]
        normed_box_trgs = torch.cat(normed_box_trgs, 0)
        normed_box_prds = box_prds / factor
        if self.loss_box_format == 'xywh':
            normed_box_prds = bbox_xyxy_to_cxcywh(normed_box_prds)
            normed_box_trgs = bbox_xyxy_to_cxcywh(normed_box_trgs)
        return normed_box_prds, normed_box_trgs

    def _calc_cls_avg_weight(
            self,
            pos_inds_list: List[Tensor],
            neg_inds_list: List[Tensor]):
        self.bg_cls_weight = 0.0 / 200  # Follow DETR.
        device = pos_inds_list[0].device
        num_total_pos = sum([inds.numel() for inds in pos_inds_list])
        num_total_neg = sum([inds.numel() for inds in neg_inds_list])
        num_total_pos = torch.as_tensor(
            [num_total_pos], dtype=torch.float, device=device)
        num_total_neg = torch.as_tensor(
            [num_total_neg], dtype=torch.float, device=device)
        num_total_pos = reduce_mean(num_total_pos)
        num_total_neg = reduce_mean(num_total_neg)
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        cls_avg_factor = max(cls_avg_factor, 1)
        return cls_avg_factor

    def _calc_box_avg_weight(self, bboxes_targets: Tensor):
        num_sampled_bboxes = bboxes_targets.any(-1).sum()
        num_sampled_bboxes = reduce_mean(
            num_sampled_bboxes.type_as(bboxes_targets))
        return num_sampled_bboxes

    def get_targets(
            self,
            cls_probs: Tensor,
            box_preds: Tensor,
            gt_bboxes_list: List[Tensor],
            gt_labels_list: List[Tensor],
            img_metas: List[Dict],
            points: List[Tensor] = None,
            gt_bboxes_ignore: List[Tensor] = None,
            num_points_per_level: List[Tensor] = None,
    ) -> Tuple[List[Tensor], ...]:
        """
        Get targets for loss calculation.

        Args:
            points: The pre-defined location coordinates.
            cls_probs: Multi-scale classification probability.
                (B, \\sum{HixWi}, C), C = num_classes.
            box_preds: Multi-scale regression predictions.
                (B, \\sum{HixWi}, 4).
            gt_bboxes_list: Multi-image bounding boxes.
                (G, 4) in format (x, y, x, y).
            gt_labels_list: Multi-image category. (G, ).
            img_metas:
        """
        self._clear_assigned_cache()
        assert gt_bboxes_ignore is None
        num_imgs = len(gt_bboxes_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_imgs)]

        (
            labels_targets_list,
            labels_weights_list,
            bboxes_targets_list,
            bboxes_weights_list,
            pos_inds_list,
            neg_inds_list,
            factor_list,
            pos_assigned_gt_inds) = multi_apply(
                self._get_one_to_one_target_single,
                cls_probs,
                box_preds,
                gt_bboxes_list,
                gt_labels_list,
                img_metas,
                points,
                gt_bboxes_ignore_list,
                num_points_per_level=num_points_per_level)
        return (
            labels_targets_list,
            labels_weights_list,
            bboxes_targets_list,
            bboxes_weights_list,
            pos_inds_list,
            neg_inds_list,
            factor_list,
            pos_assigned_gt_inds)

    def _get_one_to_one_target_single(
            self,
            cls_probs: Tensor,
            box_preds: Tensor,
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            img_meta: Dict,
            points: Tensor = None,
            gt_bboxes_ignore: Tensor = None,
            num_points_per_level: List[Tensor] = None) -> Tuple[Tensor, ...]:
        num_bboxes = box_preds.shape[0]
        img_h, img_w, _ = img_meta['img_shape']
        factor = box_preds.new_tensor(
            [img_w, img_h, img_w, img_h]).unsqueeze(0)
        sample_result: SamplingResult = self._assign_and_sample(
            self.assigner,
            cls_probs,
            box_preds,
            gt_bboxes,
            gt_labels,
            img_meta,
            points,
            gt_bboxes_ignore=gt_bboxes_ignore,
            num_points_per_level=num_points_per_level)
        pos_inds = sample_result.pos_inds
        neg_inds = sample_result.neg_inds

        labels_targets = gt_bboxes.new_full(
            (num_bboxes, ), self.num_classes, dtype=torch.long)
        labels_weights = gt_bboxes.new_ones(num_bboxes)
        bboxes_targets = torch.zeros_like(box_preds)
        bboxes_weights = torch.zeros_like(box_preds)

        labels_targets, labels_weights = self._assign_labels(
            labels_targets, labels_weights, gt_labels, sample_result)
        bboxes_targets, bboxes_weights = self._assign_bboxes(
            bboxes_targets, bboxes_weights, gt_bboxes, sample_result)
        self._positive_indices.append(pos_inds)
        self._negative_indices.append(neg_inds)
        self._assigned_indices.append(sample_result.pos_assigned_gt_inds)
        return (
            labels_targets,
            labels_weights,
            bboxes_targets,
            bboxes_weights,
            pos_inds,
            neg_inds,
            factor,
            sample_result.pos_assigned_gt_inds)

    def _assign_bboxes(
            self,
            bboxes_targets: Tensor,
            bboxes_weights: Tensor,
            gt_bboxes: Tensor,
            sample_result: SamplingResult) -> Tuple[Tensor, Tensor]:
        bboxes_targets[sample_result.pos_inds] = \
            gt_bboxes[sample_result.pos_assigned_gt_inds]
        bboxes_weights[sample_result.pos_inds] = 1.0
        return bboxes_targets, bboxes_weights

    def _assign_labels(
            self,
            labels_targets: Tensor,
            labels_weights: Tensor,
            gt_labels: Tensor,
            sample_result: SamplingResult) -> Tuple[Tensor, Tensor]:
        labels_targets[sample_result.pos_inds] = \
            gt_labels[sample_result.pos_assigned_gt_inds]
        return labels_targets, labels_weights

    def _assign_and_sample(
            self,
            assigner: BaseAssigner,
            cls_probs: Tensor,
            box_preds: Tensor,
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            img_meta: Dict,
            points: Tensor = None,
            gt_bboxes_ignore: Tensor = None,
            num_points_per_level: List[Tensor] = None) -> SamplingResult:
        img_h, img_w, _ = img_meta['img_shape']
        factor = box_preds.new_tensor(
            [img_w, img_h, img_w, img_h]).unsqueeze(0)
        normed_bbox_preds = bbox_xyxy_to_cxcywh(box_preds / factor)
        kwargs = dict()
        if isinstance(assigner, HungarianAssigner):
            pass
        elif isinstance(assigner, POTOAssigner):
            kwargs['points'] = points
            kwargs['num_points_per_level'] = num_points_per_level
        else:
            raise NotImplementedError(
                'Currently only HungarianAssigner '
                'and POTOAssigner are supported.')
        assign_result: AssignResult = assigner.assign(
            normed_bbox_preds,
            cls_probs,
            gt_bboxes,
            gt_labels,
            img_meta=img_meta,
            gt_bboxes_ignore=gt_bboxes_ignore,
            **kwargs)
        sample_result: SamplingResult = self.sampler.sample(
            assign_result, box_preds, gt_bboxes)
        return sample_result
