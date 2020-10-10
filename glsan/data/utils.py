import io
import s3path
from PIL import Image

from detectron2.data import transforms as T
import detectron2.data.detection_utils as utils

from . import transforms as T_local


def load_image_from_oss(path: s3path.S3Path, mode='rb', format=None):
    """

    Args:
        path:
        mode:
        format:

    Returns:

    """
    assert isinstance(path, s3path.S3Path)
    image = Image.open(io.BytesIO(path.open(mode=mode).read()))
    image = utils.convert_PIL_to_numpy(image, format)

    return image


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    augmentation = []
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        augmentation = [T_local.ResizeLongestEdge(min_size, max_size, sample_style)]
    if is_train:
        augmentation.append(T.RandomFlip())
    return augmentation
