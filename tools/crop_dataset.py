import sys
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import detection_utils as utils
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

from glsan.data import *
from glsan.config import get_cfg
from glsan.utils import transfrom_offsets
import numpy as np
from tqdm import tqdm
import json
import os
import multiprocessing
import torch
from sklearn.cluster import KMeans
from os import mknod, makedirs, listdir
from os.path import join, exists
import cv2
from PIL import Image, ImageDraw, ImageFont

# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
_root = '/data'

visdrone_categories = {'pedestrian': 1, 'people': 2,
                       'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6, 'tricycle': 7,
                       'awning-tricycle': 8, 'bus': 9, 'motor': 10}
uavdt_categories = {'car': 1, 'truck': 2, 'bus': 3}


def crop_dataset(cfg, dataset_name, dst_dataset_dir, cluster_num=4, categories=None, crop_size=300, padding_size=50,
                 normalized_ratio=2):
    dataset = DatasetCatalog.get(dataset_name)
    image_id = 1
    bbox_id = 1

    # create paths
    dst_dataset_dir = join(_root, dst_dataset_dir)
    annotation_dir = join(dst_dataset_dir, 'annotations')
    train_img_dir = join(dst_dataset_dir, 'train')
    if not exists(dst_dataset_dir):
        makedirs(dst_dataset_dir)
        makedirs(train_img_dir)
        makedirs(annotation_dir)

    json_dict = {'images': [], 'type': 'instances', 'categories': [], 'annotations': []}
    cluster_rate = 0
    cluster_images = 0
    for dataset_dict in tqdm(dataset):
        image = utils.read_image(dataset_dict["file_name"], format=cfg.INPUT.FORMAT)
        image_shape = image.shape[:2]
        annos = [
            transform_box_mode(obj, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=cfg.INPUT.MASK_FORMAT
        )
        instances = utils.filter_empty_instances(instances)
        dataset_dict['instances'] = instances

        # compute boxes for cropping
        boxes = instances.gt_boxes.tensor.numpy().astype(np.int32)
        gt_classes = instances.gt_classes.numpy()
        points = np.stack(((boxes[:, 0] + boxes[:, 2]) / 2,
                           (boxes[:, 1] + boxes[:, 3]) / 2),
                          axis=1)
        sizes = []
        if len(points) < cluster_num and crop_size >= 0:
            centers = [[] for i in range(len(points))]
            sizes = [[] for i in range(len(points))]
            ranges = [[] for i in range(len(points))]  # [x,y,x,y]
            lbs = np.arange(len(points))
            for i in range(len(points)):
                min_w, min_h, max_w, max_h = boxes[i]
                max_height = max_h - min_h + padding_size
                max_width = max_w - min_w + padding_size
                sizes[i].append(max([max_height, max_width // normalized_ratio, crop_size]))
                sizes[i].append(max([max_height // normalized_ratio, max_width, crop_size]))
                centers[i] = [(min_h + max_h) // 2, (min_w + max_w) // 2]
                ranges[i] = [max(0, centers[i][1] - sizes[i][1] / 2),
                             max(0, centers[i][0] - sizes[i][0] / 2),
                             min(image_shape[1], centers[i][1] + sizes[i][1] / 2),
                             min(image_shape[0], centers[i][0] + sizes[i][0] / 2)]
        else:
            cluster_images += 1
            centers = [[] for i in range(cluster_num)]
            sizes = [[] for i in range(cluster_num)]
            ranges = [[] for i in range(cluster_num)]  # [x,y,x,y]
            kmeans = KMeans(n_clusters=cluster_num)
            classes = kmeans.fit(points)
            lbs = classes.labels_

            for i in range(cluster_num):
                boxes_class_i = boxes[lbs == i]
                boxes_class_i = boxes_class_i.reshape(-1, 2)
                min_w, min_h = boxes_class_i.min(0)
                max_w, max_h = boxes_class_i.max(0)
                max_height = max_h - min_h + padding_size
                max_width = max_w - min_w + padding_size
                sizes[i].append(max([max_height, max_width // normalized_ratio, crop_size]))
                sizes[i].append(max([max_height // normalized_ratio, max_width, crop_size]))
                centers[i] = [(min_h + max_h) // 2, (min_w + max_w) // 2]
                ranges[i] = [max(0, centers[i][1] - sizes[i][1] / 2),
                             max(0, centers[i][0] - sizes[i][0] / 2),
                             min(image_shape[1], centers[i][1] + sizes[i][1] / 2),
                             min(image_shape[0], centers[i][0] + sizes[i][0] / 2)]
        ranges = np.asarray(ranges).astype(np.int32)

        # img = Image.fromarray(image)
        # draw = ImageDraw.Draw(img)
        # for range_i in range(len(ranges)):
        #     r = ranges[range_i]
        #     draw.rectangle(r, outline=(255, 0, 0))
        # img.save('crop.jpg')

        # crop image
        file_name = dataset_dict['file_name']
        file_head = file_name.split('/')[-1].split('.')[0]
        parent_dir = file_name.split('/')[-2]

        # save original image
        img_name = file_head + '.jpg'
        if parent_dir[0] == 'M':
            if not exists(join(train_img_dir, parent_dir)):
                makedirs(join(train_img_dir, parent_dir))
            img_name = join(parent_dir, img_name)
        ori_file_name = join(train_img_dir, img_name)
        cv2.imwrite(ori_file_name, image)

        image_dict = {'file_name': img_name, 'height': image_shape[0], 'width': image_shape[1],
                      'id': image_id}
        json_dict['images'].append(image_dict)
        ori_boxes = boxes.tolist()
        ori_gt_classes = gt_classes.tolist()

        for obj_i in range(len(ori_boxes)):
            box_i = ori_boxes[obj_i]
            ori_gt_classes_i = ori_gt_classes[obj_i]
            category_id = ori_gt_classes_i + 1  # id index start from 1
            o_width = box_i[2] - box_i[0]
            o_height = box_i[3] - box_i[1]
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                   'bbox': [box_i[0], box_i[1], o_width, o_height], 'category_id': category_id,
                   'id': bbox_id, 'ignore': 0, 'segmentation': []}
            bbox_id += 1
            json_dict['annotations'].append(ann)
        image_id += 1

        # save cropped images
        for range_i in range(len(sizes)):
            # r = transfrom_offsets(centers[i], sizes[i], image_shape[0], image_shape[1])  # [x,y,x,y]
            r = ranges[range_i]
            sub_image = image[r[1]:r[3], r[0]:r[2]]
            sub_image_shape = sub_image.shape[:2]
            sub_img_name = file_head + '_' + str(range_i) + '.jpg'
            if parent_dir[0] == 'M':
                sub_img_name = join(parent_dir, sub_img_name)
            sub_file_name = join(train_img_dir, sub_img_name)
            cv2.imwrite(sub_file_name, sub_image)

            image_dict = {'file_name': sub_img_name, 'height': sub_image_shape[0], 'width': sub_image_shape[1],
                          'id': image_id}
            json_dict['images'].append(image_dict)
            offset = np.tile(r[0:2], 2)
            sub_boxes = (boxes[lbs == range_i] - offset).astype(np.int32).tolist()
            sub_gt_classes = gt_classes[lbs == range_i].tolist()

            for obj_i in range(len(sub_boxes)):
                box_i = sub_boxes[obj_i]
                sub_gt_classes_i = sub_gt_classes[obj_i]
                category_id = sub_gt_classes_i + 1  # id index start from 1
                o_width = box_i[2] - box_i[0]
                o_height = box_i[3] - box_i[1]
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                       'bbox': [box_i[0], box_i[1], o_width, o_height], 'category_id': category_id,
                       'id': bbox_id, 'ignore': 0, 'segmentation': []}
                bbox_id += 1
                json_dict['annotations'].append(ann)
            image_id += 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_file = join(annotation_dir, 'train.json')
    if not exists(json_file):
        mknod(json_file)
    json.dump(json_dict, open(json_file, 'w'), indent=4)
    print(cluster_images)
    print(cluster_images/len(dataset))


def transform_box_mode(annotation, image_shape):
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    annotation["bbox"] = np.minimum(bbox, list(image_shape + image_shape)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def dataset_len(image_path):
    count = 0
    if 'UAVDT' in image_path:
        seqs = listdir(image_path)
        for seq in seqs:
            count += len(listdir(join(image_path,seq)))
    else:
        return listdir(image_path)
    return count


if __name__ == "__main__":
    print(dataset_len('/data/UAVDTPlus/train'))
    # args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # cfg = setup(args)
    # crop_dataset(cfg, 'visdrone_train', 'VisDronePlus0', cluster_num=4, categories=visdrone_categories,
    #              crop_size=0)
