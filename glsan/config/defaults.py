from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg as d2config

_C = d2config()

# ---------------------------------------------------------------------------- #
# GLSAN Options
# ---------------------------------------------------------------------------- #
_C.GLSAN = CN()
_C.GLSAN.CROP = "NoCrop"  # options: "NoCrop", "UniformlyCrop", "SelfAdaptiveCrop"
_C.GLSAN.CLUSTER_NUM = 4
_C.GLSAN.CROP_SIZE = 300
_C.GLSAN.PADDING_SIZE = 50
_C.GLSAN.NORMALIZED_RATIO = 2.0  # >=1
_C.GLSAN.SR = False
_C.GLSAN.SR_THRESH = 500
