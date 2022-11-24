
"""
CrowdHuman dataset.

Author:
    Yiqun Chen
"""

from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.builder import DATASETS

from date.datasets.utils import (
    format_dt_json,
    format_gt_json,
    evaluate_crowdhuman)


@DATASETS.register_module()
class CrowdHumanDataset(CocoDataset):

    CLASSES = ('person', )

    PALETTE = [(220, 20, 60)]

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids()

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Keep the API consistent
                with mmdet, not used.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Keep the API consistent with mmdet, not used.
            proposal_nums (Sequence[int]): Keep the API consistent
                with mmdet, not used.
            iou_thrs (Sequence[float], optional): Keep the API consistent
                with mmdet, not used.
            metric_items (list[str] | str, optional): Keep the API consistent
                with mmdet, not used.

        Returns:
            dict[str, float]: CrowdHuman evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids()

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        path2dt = format_dt_json(tmp_dir.name, result_files['bbox'])
        path2gt = format_gt_json(tmp_dir.name, self.ann_file)
        eval_results = evaluate_crowdhuman(path2gt, path2dt)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
