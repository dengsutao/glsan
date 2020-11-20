import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from .builtin_meta import _get_visdrone_instances_meta, _get_uavdt_instances_meta


def register_coco_instances(name, metadata, json_file, image_root):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_all_visdrone(root):
    SPILTS = {
        "visdrone_train": (
            "VisDrone2019-DET/train",
            "VisDrone2019-DET/annotations/train.json"
        ),
        "visdrone_val": (
            "VisDrone2019-DET/val",
            "VisDrone2019-DET/annotations/val.json"
        ),
        # merge cropped sub_images into original dataset for training
        "visdroneplus_train": (
            "VisDronePlus/train",
            "VisDronePlus/annotations/train.json"
        ),
        "visdronepp_train": (
            "VisDronePP/train",
            "VisDronePP/annotations/train.json"
        ),
        "visdroneplus600_train": (
            "VisDronePlus600/train",
            "VisDronePlus600/annotations/train.json"
        ),
        "visdroneplus300_train": (
            "VisDronePlus300/train",
            "VisDronePlus300/annotations/train.json"
        ),
        "visdroneplus400_train": (
            "VisDronePlus400/train",
            "VisDronePlus400/annotations/train.json"
        ),
        "visdroneplus500_train": (
            "VisDronePlus500/train",
            "VisDronePlus500/annotations/train.json"
        ),
        "visdroneplus0_train": (
            "VisDronePlus0/train",
            "VisDronePlus0/annotations/train.json"
        ),
        "visdronepp300_train": (
            "VisDronePP300/train",
            "VisDronePP300/annotations/train.json"
        ),
        "visdronepp0_train": (
            "VisDronePP0/train",
            "VisDronePP0/annotations/train.json"
        ),
        "visdroneplus200_train": (
            "VisDronePlus200/train",
            "VisDronePlus200/annotations/train.json"
        ),
        "visdronepp200_train": (
            "VisDronePP200/train",
            "VisDronePP200/annotations/train.json"
        ),
        "visdrone_val_common": (
            "VisDrone-val-common/images",
            "VisDrone-val-common/annotations/val.json"
        ),
        "visdrone_val_sparse": (
            "VisDrone-val-sparse/images",
            "VisDrone-val-sparse/annotations/val.json"
        ),
        "visdrone_val_dense": (
            "VisDrone-val-dense/images",
            "VisDrone-val-dense/annotations/val.json"
        ),
    }
    for key, (image_root, json_file) in SPILTS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_visdrone_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_uavdt(root):
    SPILTS = {
        "uavdt_train": (
            "UAVDT/train",
            "UAVDT/annotations/train.json"
        ),
        "uavdt_val": (
            "UAVDT/val",
            "UAVDT/annotations/val.json"
        ),
        "uavdtplus_train": (
            "UAVDTPlus/train",
            "UAVDTPlus/annotations/train.json"
        ),
    }
    for key, (image_root, json_file) in SPILTS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_uavdt_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_visdrone(_root)
register_all_uavdt(_root)
