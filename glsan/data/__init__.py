from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog, Metadata
# ensure the builtin datasets are registered
from . import datasets
