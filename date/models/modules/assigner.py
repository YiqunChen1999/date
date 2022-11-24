
"""
Assigner to support one-to-many matching via multi-one-to-one fashion.

Author:
    Yiqun Chen
"""

from typing import Dict, List

from scipy.optimize import linear_sum_assignment
import torch
from torch import Tensor
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from date.utils.grid import get_points_in_center_area_mask


@BBOX_ASSIGNERS.register_module()
class POTOAssigner(BaseAssigner):
    """
    Prediction-aware One-to-One (POTO) label assigment strategy.

    Firstly described in
    `End-to-End Object Detection with Fully Convolutional Network`.
    arxiv: https://arxiv.org/abs/2012.03544

    Args:
        alpha: Controling the geometric weight.
        iou: The type of IoU type used in formula.
    """
    def __init__(
            self,
            alpha: float = 0.8,
            iou_type: str = 'giou',
            strides: List[int] = [8, 16, 32, 64, 128],
            center_sampling_radius: float = 1.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.iou_type = iou_type
        self.strides = strides
        self.center_sampling_radius = center_sampling_radius

    def assign(self,
               box_preds: Tensor,
               cls_preds: Tensor,
               gt_bboxes: Tensor,
               gt_labels: Tensor,
               img_meta: Dict,
               points: Tensor,
               num_points_per_level: List[int],
               gt_bboxes_ignore: Tensor = None,
               eps: float = 1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_preds (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. This design
                is compatible with HungarianAssigner. Shape: [N, 4].
            cls_preds (Tensor): Predicted classification logits, shape
                [N, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). This design is
                compatible with HungarianAssigner. Shape: [G, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (G,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_points = len(points)
        num_gts = len(gt_bboxes)
        inside_gt_mask = get_points_in_center_area_mask(
            gt_bboxes,
            points,
            self.strides,
            self.center_sampling_radius,
            num_points_per_level)
        assert inside_gt_mask.shape == (num_points, num_gts), \
               f'{inside_gt_mask.shape} == {(num_points, num_gts)}'

        cls_quality = self._get_cls_quality(cls_preds, gt_labels)
        assert cls_quality.shape == inside_gt_mask.shape, \
               f'{cls_quality.shape} == {inside_gt_mask.shape}'

        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor(
            [img_w, img_h, img_w, img_h]).unsqueeze(0)
        box_preds = bbox_cxcywh_to_xyxy(box_preds) * factor
        box_quality = self._get_box_quality(box_preds, gt_bboxes)
        assert box_quality.shape == inside_gt_mask.shape, \
               f'{box_quality.shape} == {inside_gt_mask.shape}'

        # Quality, the higher the better.
        quality = cls_quality * box_quality * inside_gt_mask

        matched_row_inds, matched_col_inds = \
            linear_sum_assignment(quality.detach().cpu(), maximize=True)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            box_preds.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            box_preds.device)

        num_bboxes = box_preds.size(0)
        assigned_gt_inds = box_preds.new_full(
            (num_bboxes, ), -1, dtype=torch.long)
        assigned_labels = box_preds.new_full(
            (num_bboxes, ), -1, dtype=torch.long)
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        num_gts = gt_bboxes.size(0)
        assign_result = AssignResult(
            num_gts, assigned_gt_inds, None, assigned_labels)
        return assign_result

    def _get_box_quality(
            self, box_preds: Tensor, gt_bboxes: Tensor) -> Tensor:
        iou = bbox_overlaps(box_preds, gt_bboxes)
        return iou ** self.alpha

    def _get_cls_quality(
            self, cls_preds: Tensor, gt_labels: Tensor) -> Tensor:
        return cls_preds[:, gt_labels].sigmoid() ** (1 - self.alpha)
