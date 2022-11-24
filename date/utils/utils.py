
from typing import List, Tuple
from warnings import warn

import torch
from torch import nn, Tensor


def sort_and_keep_topk_results(results: List[List[Tensor]], topk: int):
    return [sort_and_keep_topk_results_single_image(r, topk) for r in results]


def sort_and_keep_topk_results_single_image(result: List[Tensor], topk: int):
    bboxes, scores, labels = result
    scores, inds = scores.sort(descending=True)
    bboxes = bboxes[inds]
    labels = labels[inds]
    scores = scores[:topk]
    bboxes = bboxes[:topk]
    labels = labels[:topk]
    return torch.cat([bboxes, scores[:, None]], -1), labels
