VISDRONE_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "pedestrian"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "people"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "bicycle"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "car"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "van"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "truck"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "tricycle"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "awning-tricycle"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "bus"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "motor"}
]

UAVDT_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "car"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "truck"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "bus"}
]


def _get_visdrone_instances_meta():
    thing_ids = [k["id"] for k in VISDRONE_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VISDRONE_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 10, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 9]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VISDRONE_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_uavdt_instances_meta():
    thing_ids = [k["id"] for k in UAVDT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in UAVDT_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 3, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 2]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in UAVDT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret
