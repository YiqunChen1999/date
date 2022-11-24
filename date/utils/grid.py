
from typing import List, Dict, Tuple
import torch
from torch import Tensor

from mmdet.core import multi_apply
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh


def get_points_in_boxes_mask(points: Tensor, boxes: Tensor) -> Tensor:
    """
    Get a mask indicates whether a point is fall in a box.

    Args:
        points: Coordinates in format (x, y) with shape (N, 2)
        boxes: Coordinates in format (x, y, x, y) with shape (G, 4)

    Returns:
        fall_in_boxes: A mask with shape (N, G)
            indicates whether a point is fall in a box.

    NOTE It's user's responsibility to ensure both points and boxes are
    normalized or unnormalized.
    """
    points = points.unsqueeze(1).repeat(1, len(boxes), 1)  # (N, G, 2)
    boxes = boxes.unsqueeze(0).repeat(len(points), 1, 1)  # (N, G, 4)
    fall_in_boxes = \
        (points[..., 0] > boxes[..., 0]) \
        & (points[..., 1] > boxes[..., 1]) \
        & (points[..., 0] < boxes[..., 2]) \
        & (points[..., 1] < boxes[..., 3])
    return fall_in_boxes


def get_points_in_center_area_mask(
        gt_bboxes: Tensor,
        points: Tensor,
        strides: List[int],
        center_sampling_radius: int,
        num_points_per_level: List[int]) -> Tensor:
    """
    Get the mask which indicates whether a point is
    in the area of a center of gt bbox or not.

    NOTE The format of gt_bboxes and points should be both normalized
        or unnormalized, and it's user's responsibility.

    Args:
        gt_bboxes: Ground truth bounding boxes
            in (x, y, x, y) format with shape (G, 4).
        points: points in (x, y) format with shape (N, 2). N is the total
            number of points, which sums over all feature levels.
        strides: A list records the stride of each feature level.
        center_sampling_radius: Points within the radius of
            center_sampling_radius*stride in absolute coordinate will
            be treated as positive.
        num_points_per_level: Numbers of points of each feature level.

    Returns:
        inside_gt_bbox_mask: A mask with shape (N, G),
            indicate whether a point falls into the center area of a gt bbox.
    """
    num_points = points.size(0)
    num_gts = gt_bboxes.size(0)
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
    xs, ys = points[:, 0], points[:, 1]
    xs = xs[:, None].expand(num_points, num_gts)
    ys = ys[:, None].expand(num_points, num_gts)

    # condition1: inside a `center bbox`
    radius = center_sampling_radius
    center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
    center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
    center_gts = torch.zeros_like(gt_bboxes)
    stride = center_xs.new_zeros(center_xs.shape)

    # project the points on current lvl back to the `original` sizes
    lvl_begin = 0
    for lvl_idx, num_points_lvl in enumerate(num_points_per_level):
        lvl_end = lvl_begin + num_points_lvl
        stride[lvl_begin:lvl_end] = strides[lvl_idx] * radius
        lvl_begin = lvl_end

    x_mins = center_xs - stride
    y_mins = center_ys - stride
    x_maxs = center_xs + stride
    y_maxs = center_ys + stride
    center_gts[..., 0] = torch.where(
        x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
    center_gts[..., 1] = torch.where(
        y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
    center_gts[..., 2] = torch.where(
        x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs)
    center_gts[..., 3] = torch.where(
        y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs)

    cb_dist_left = xs - center_gts[..., 0]
    cb_dist_right = center_gts[..., 2] - xs
    cb_dist_top = ys - center_gts[..., 1]
    cb_dist_bottom = center_gts[..., 3] - ys
    center_bbox = torch.stack(
        (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

    return inside_gt_bbox_mask
