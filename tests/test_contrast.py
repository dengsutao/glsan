import fire
import torch
import torch.nn as nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

from slender_det.config import get_cfg
from slender_det.modeling import build_model
from slender_det.data import build_detection_train_loader

# get cfg
cfg = get_cfg()
cfg.merge_from_file("configs/contrast/Base.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2

# get batch data
data_loader = build_detection_train_loader(cfg)


def test_training():
    setup_logger(name="slender_det")

    model = build_model(cfg)

    device = torch.device("cuda")
    model.to(device)

    for batch in data_loader:
        model(batch)
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    fire.Fire()
