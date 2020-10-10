import logging 
import torch

from detectron2.data import (
    build_detection_test_loader,
    get_detection_dataset_dicts,
    build_batch_data_loader)
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import RepeatFactorTrainingSampler, TrainingSampler

from slender_det.evaluation.coco import COCO


from . import mappers

def repeat_factors_from_ratios(dataset_dicts):
    rep_factors = []
    for dataset_dict in dataset_dicts:
        rep_factor = 0.1
        for ann in dataset_dict["annotations"]:
            ratio = COCO.compute_ratio(ann)["ratio"]
            if ratio < 1/5:
                rep_factor = 1
                break
            if ratio < 1/3:
                rep_factor = 0.5
        rep_factors.append(rep_factor)
    return torch.tensor(rep_factors, dtype=torch.float32)


def get_dataset_mapper(dataset_name):
    if "coco" in dataset_name:
        return getattr(mappers, "DatasetMapper", None)
    elif "objects365" in dataset_name:
        return getattr(mappers, "OssMapper", None)
    else:
        return getattr(mappers, "DatasetMapper", None)


def build_train_loader(cfg, mapper=None):
    if mapper is None:
        mapper = get_dataset_mapper(cfg.DATASETS.TRAIN[0])(cfg, True)

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    elif sampler_name == "RatioFactorTrainingSampler":
        repeat_factors = repeat_factors_from_ratios(
            dataset_dicts
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)

    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_test_loader(cfg, dataset_name, mapper=None):
    if mapper is None:
        mapper = get_dataset_mapper(dataset_name)(cfg, False)

    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
