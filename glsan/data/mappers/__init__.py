# from detectron2.data.dataset_mapper import DatasetMapper

from .base import DatasetMapper
from .bm_mapper import BorderMaskMapper
from .oss_mapper import OssMapper, load_image_from_oss
