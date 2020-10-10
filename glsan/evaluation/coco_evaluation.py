import os
import copy
import itertools
from collections import OrderedDict
import json
import numpy as np
import contextlib
import io
import logging
import pickle
import pycocotools.mask as mask_util
from tabulate import tabulate

import torch
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.evaluation import COCOEvaluator as Base
from detectron2.utils.logger import create_small_table

from .cocoeval import COCOeval
from .coco import COCO
from concern.support import between_ranges


class COCOEvaluator(Base):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
                " Trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        # with contextlib.redirect_stdout(io.StringIO()):
        self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def evaluate(self, name="coco"):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            PathManager.mkdirs(os.path.join(self._output_dir, name))
            file_path = os.path.join(self._output_dir, name, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._evaluate_predictions_ar(predictions)
            self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
                Evaluate predictions on the given tasks.
                Fill self._results with the metrics of the tasks.
                """
        print("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                        category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            print("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            print("Annotations are not available for evaluation.")
            return

        print("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        print(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            print("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_per_category_r = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            results_per_category_r.append(
                ("{}".format(name),
                precisions[:, :, idx, :, -1].mean(0).mean(0)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        print("Per-category {} AP: \n".format(iou_type) + table)

        results["AP-ratios"] = {"AP-" + name: ap.tolist() for name, ap in results_per_category_r}
        return results


    def _evaluate_predictions_ar(self, predictions):
        res = {}
        aspect_ratios = {
            "all ratios": [0 / 1, 1e5 / 1],
            " 0  - 1/5":  [0 / 1, 1 / 5],
            "1/5 - 1/3":  [1 / 5, 1 / 3],
            "1/3 - 3/1":  [1 / 3, 3 / 1],
            "3/1 - 5/1":  [3 / 1, 5 / 1],
            "5/1 - INF":  [5 / 1, 1e5 / 1],
        }
        areas = {
            "all areas": [0, float("inf")],
            "small":     [0, 32**2],
            "medium":    [32**2, 96**2],
            "large":     [96**2, float("inf")]
        }
        limits = [100]
        for limit in limits:
            stats = _evaluate_predictions_ar(
                predictions,
                self._coco_api,
                self._metadata,
                aspect_ratios=aspect_ratios,
                areas=areas,
                limit=limit)
            recalls = stats.pop("recalls")
            for i, key in enumerate(areas):
                res["AR-{}@{:d}".format(key, limit)] = recalls[:, -1, 0, i].mean() * 100
                res["mAR-{}@{:d}".format(key, limit)] = recalls[:, :-1, 0, i].mean() * 100

            for i, key in enumerate(aspect_ratios):
                res["AR-{}@{:d}".format(key, limit)] = recalls[:, -1, i, 0].mean() * 100
                res["mAR-{}@{:d}".format(key, limit)] = recalls[:, :-1, i, 0].mean() * 100

            key = "AR@{:d}".format(limit)
            res[key] = float(stats["ar"].item() * 100)
            key = "mAR@{:d}".format(limit)
            res[key] = float(stats["mar"].item() * 100)

        print("Proposal metrics: \n" + create_small_table(res))
        # stats["recalls"] = recalls
        res["ar-stats"] = stats
        self._results["ar"] = res


def _evaluate_predictions_ar(
        predictions,
        coco_api,
        metadata,
        thresholds=None,
        aspect_ratios={},
        areas={},
        limit=None):
    cats = coco_api.cats.values()
    ratios = list(aspect_ratios.values())
    areas = list(areas.values())
    K = len(cats) + 1  # -1 for all classes
    R = len(ratios)
    A = len(areas) # Area ranges
    
    counts_matrixes = []
    overlap_matrixes = []

    gt_overlaps = []

    for prediction_dict in predictions:
        count_matrix = torch.zeros((K, R, A), dtype=torch.int32)

        image_id = prediction_dict["image_id"]
        predictions = prediction_dict["instances"]
        predict_boxes = [
            BoxMode.convert(prediction['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for prediction in predictions
        ]
        predict_classes = torch.tensor([
            prediction["category_id"] for prediction in predictions
        ])
        predict_boxes = torch.as_tensor(predict_boxes).reshape(-1, 4)
        predict_boxes = Boxes(predict_boxes)

        ann_ids = coco_api.getAnnIds(imgIds=image_id)
        anno = coco_api.loadAnns(ann_ids)
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
        ]
        gt_classes = torch.tensor([
            metadata.thing_dataset_id_to_contiguous_id[obj["category_id"]]
            for obj in anno])
        gt_aspect_ratios = [
            obj["ratio"] for obj in anno
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_aspect_ratios = torch.as_tensor(gt_aspect_ratios)
        gt_areas = torch.as_tensor(
            [(box[2] - box[0]) * (box[3] - box[1]) for box in gt_boxes])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        if len(gt_boxes) == 0:
            continue

        N = len(gt_boxes)
        overlap_matrix = torch.zeros((K, R, A, N), dtype=torch.float32)
        for i in range(len(gt_boxes)):
            k = gt_classes[i]
            r = between_ranges(gt_aspect_ratios[i], ratios)
            a = torch.tensor(between_ranges(gt_areas[i], areas)).nonzero()
            count_matrix[k, r, a] += 1
            count_matrix[-1, r, a] += 1

        if limit is not None and len(predictions) > limit:
            predict_boxes = predict_boxes[:limit]

        overlaps = pairwise_iou(predict_boxes, gt_boxes)
        class_matched = predict_classes[:, None] == gt_classes[None]
        overlaps_when_matched = overlaps * class_matched

        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            max_overlaps_m, argmax_overlaps_m = overlaps_when_matched.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            gt_ovr_m, gt_ind_m = max_overlaps_m.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            box_ind_m = argmax_overlaps_m[gt_ind_m]
            # record the iou coverage of this gt box
            k = gt_classes[gt_ind_m]
            r = between_ranges(gt_aspect_ratios[gt_ind_m], ratios)
            a = torch.tensor(between_ranges(gt_areas[gt_ind_m], areas)).nonzero()
            n = (torch.arange(N) == j).nonzero()
            overlap_matrix[k, r, a, n] = overlaps_when_matched[box_ind_m, gt_ind_m]
            overlap_matrix[-1, r, a, n] = overlaps[box_ind, gt_ind]
            assert torch.all(overlap_matrix[k, r, a, n] == gt_ovr_m)
            assert torch.all(overlap_matrix[-1, r, a, n] == gt_ovr)
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
            overlaps_when_matched[box_ind_m, :] = -1
            overlaps_when_matched[:, gt_ind_m] = -1

        # append recorded iou coverage level
        overlap_matrixes.append(overlap_matrix)
        counts_matrixes.append(count_matrix)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    T = len(thresholds)
    recalls = torch.zeros((T, K, R, A))

    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        count = torch.zeros((K, R, A))
        hit = torch.zeros((K, R, A))
        for count_matrix, overlap_matrix in zip(counts_matrixes, overlap_matrixes):
            hit_matrix = (overlap_matrix >= t).float().sum(-1)
            count += count_matrix
            hit += hit_matrix
        recalls[i] = hit / torch.max(
            count.float(), torch.tensor(1).float())
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls[:, -1, 0, 0].mean()
    mar = recalls[:, :-1, 0, 0].mean()
    return {
        "ar": ar,
        "mar": mar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": torch.stack(counts_matrixes).sum(0),
    }


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

    if iou_type == "keypoints":
        num_keypoints = len(coco_results[0]["keypoints"]) // 3
        assert len(coco_eval.params.kpt_oks_sigmas) == num_keypoints, (
            "[COCOEvaluator] The length of cfg.TEST.KEYPOINT_OKS_SIGMAS (default: 17) "
            "must be equal to the number of keypoints. However the prediction has {} "
            "keypoints! For more information please refer to "
            "http://cocodataset.org/#keypoints-eval.".format(num_keypoints)
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
