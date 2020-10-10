from .defaults import _C
from detectron2.config import CfgNode as CN


def get_cfg() -> CN:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C
