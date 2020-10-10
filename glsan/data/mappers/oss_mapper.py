import copy
import os
import io
import s3path
import logging
import numpy as np
from PIL import Image
from concern.smart_path import smart_path
import torch

from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils


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


class OssMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.augmentations
        self.oss_root = cfg.DATALOADER.OSS_ROOT

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        file_path = os.path.join(self.oss_root, dataset_dict['file_name'])
        image = load_image_from_oss(smart_path(file_path), format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)

        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            # add crowd flag for each box
            iscrowd = [obj.get("iscrowd", 0) for obj in dataset_dict["annotations"]]
            iscrowd = torch.tensor(iscrowd, dtype=torch.int32)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
            ]

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            # is safe to add crowd to _fields of Instance
            # __getitem__ method will index all value in Instance._fields dict
            instances.iscrowd = iscrowd
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            # For load objects365 dataset and predict mask, we skip getting bounding boxes from safe annos
            # if self.recompute_boxes and instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
