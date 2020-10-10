from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg as d2config

_C = d2config()

# -------------------------------------------------------------------------------#
# Required by PSO
# -------------------------------------------------------------------------------#

_C.DEBUG = False

_C.USE_NORI = False
_C.NEED_MASKS = False

_C.DATALOADER.OSS_ROOT = "s3://detection/"
_C.NORI_PATH = "s3://detection/datasets/coco/"
_C.REDIS = CN()
_C.REDIS.HOST = "10.124.171.195"
_C.REDIS.PORT = 6379
_C.REDIS.DB = 0

# ---------------------------------------------------------------------------- #
# Contrast Det Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CONTRAST = CN()
_C.MODEL.CONTRAST.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
_C.MODEL.CONTRAST.STACK_CONVS = 3
_C.MODEL.CONTRAST.FEAT_CHANNELS = 256
_C.MODEL.CONTRAST.NORM_MODE = "GN"
_C.MODEL.CONTRAST.EMBED_CHANNELS = 128

