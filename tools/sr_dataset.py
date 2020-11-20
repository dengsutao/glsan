import numpy as np
from tqdm import tqdm
import json
import os
import torch
from os import mknod, makedirs
from os.path import join, exists
import cv2
import math
from glsan.modeling import EDSR
from .visualizer import visualize_gt_boxes


@torch.no_grad()
def crop2sr(crop_dataset_dir, sr_dataset_dir, sr_model, dataset_type="visdrone", sr_size=300):
    sr_img_dir = join(sr_dataset_dir, 'train')
    sr_anno_dir = join(sr_dataset_dir, 'annotations')
    crop_img_dir = join(crop_dataset_dir, 'train')
    sr_json_file = join(sr_anno_dir, 'train.json')
    if not exists(sr_dataset_dir):
        makedirs(sr_dataset_dir)
        makedirs(sr_img_dir)
        makedirs(sr_anno_dir)
        mknod(join(sr_json_file))
    crop_json_file = join(crop_dataset_dir, 'annotations', 'train.json')
    crop_json = json.loads(open(crop_json_file, 'r').read())

    anno_ind = 0
    len_anno = len(crop_json['annotations'])
    for img_i in tqdm(range(len(crop_json['images']))):
        img_info = crop_json['images'][img_i]
        img = cv2.imread(join(crop_img_dir, img_info['file_name']))
        is_sr = False
        if math.sqrt(img_info['height'] * img_info['width']) < sr_size:
            is_sr = True
            sr_input = torch.FloatTensor(img.copy()).to(device).permute(2, 0, 1).unsqueeze(0)
            img = sr_model(sr_input)
            img = img.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img_info['height'] = img_info['height'] * 2
            img_info['width'] = img_info['width'] * 2
        # boxes = []
        # classes = []
        while anno_ind < len_anno and crop_json['annotations'][anno_ind]['image_id'] == img_info['id']:
            anno = crop_json['annotations'][anno_ind]
            if is_sr:
                anno['area'] = anno['area'] * 4
                anno['bbox'] = [x * 2 for x in anno['bbox']]  # anno['bbox']是一个list!!!
            # boxes.append([anno['bbox'][0], anno['bbox'][1],
            #               anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]])
            # classes.append(anno['category_id'])
            anno_ind += 1
        sr_img_file = join(sr_img_dir, img_info['file_name'])
        if not exists(sr_img_file):
            cv2.imwrite(sr_img_file, img)
    json.dump(crop_json, open(sr_json_file, 'w'), indent=4)


@torch.no_grad()
def crop_add_sr(crop_dataset_dir, sr_dataset_dir, sr_model, dataset_type="visdrone", sr_size=500):
    sr_img_dir = join(sr_dataset_dir, 'train')
    sr_anno_dir = join(sr_dataset_dir, 'annotations')
    crop_img_dir = join(crop_dataset_dir, 'train')
    sr_json_file = join(sr_anno_dir, 'train.json')
    if not exists(sr_dataset_dir):
        makedirs(sr_dataset_dir)
        makedirs(sr_img_dir)
        makedirs(sr_anno_dir)
        mknod(join(sr_json_file))
    crop_json_file = join(crop_dataset_dir, 'annotations', 'train.json')
    crop_json = json.loads(open(crop_json_file, 'r').read())

    sr_json = {'images': [], 'type': 'instances', 'categories': crop_json['categories'], 'annotations': []}
    sr_image_id = 1
    sr_bbox_id = 1
    anno_ind = 0
    len_anno = len(crop_json['annotations'])
    for img_i in tqdm(range(len(crop_json['images']))):
        img_info = crop_json['images'][img_i]
        img = cv2.imread(join(crop_img_dir, img_info['file_name']))

        # add original crop dataset
        anno_ind_start = anno_ind
        while anno_ind < len_anno and crop_json['annotations'][anno_ind]['image_id'] == img_info['id']:
            anno = crop_json['annotations'][anno_ind]
            anno['bbox_id'] = sr_bbox_id
            sr_bbox_id += 1
            anno_ind += 1
            sr_json['annotations'].append(anno)
        anno_ind_end = anno_ind

        img_info['id'] = sr_image_id
        sr_json['images'].append(img_info)
        sr_image_id += 1
        crop_img_file = join(sr_img_dir, img_info['file_name'])
        if not exists(crop_img_file):
            cv2.imwrite(crop_img_file, img)

        # add sr dataset
        is_sr = False
        if math.sqrt(img_info['height'] * img_info['width']) < sr_size:
            is_sr = True
            sr_input = torch.FloatTensor(img.copy()).to(device).permute(2, 0, 1).unsqueeze(0)
            img = sr_model(sr_input)
            img = img.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        anno_ind = anno_ind_start
        if is_sr:
            while anno_ind < anno_ind_end:
                anno = crop_json['annotations'][anno_ind]
                sr_anno = {'area': anno['area'] * 4, 'iscrowd': 0, 'image_id': sr_image_id,
                           'bbox': [x * 2 for x in sr_anno['bbox']],
                           'category_id': anno['category_id'],
                           'id': sr_bbox_id, 'ignore': 0, 'segmentation': []}
                sr_json['annotations'].append(sr_anno)
                sr_bbox_id += 1
                anno_ind += 1
            sr_dict = {'file_name': img_info['file_name'].split('.')[0] + 's.jpg',
                       'height': img_info['height'] * 2,
                       'width': img_info['width'] * 2,
                       'id': sr_image_id}
            sr_json['images'].append(sr_dict)
            sr_image_id += 1
            sr_img_file = join(sr_img_dir, sr_dict['file_name'])
            if not exists(sr_img_file):
                cv2.imwrite(sr_img_file, img)
    json.dump(sr_json, open(sr_json_file, 'w'), indent=4)


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


device = 'cuda'
if __name__ == "__main__":
    sr_model = EDSR().to(device)
    sr_model.load_state_dict(torch.load('./models/visdrone_x2.pt'))
    crop_add_sr('/data/VisDronePlus', '/data/VisDronePPP', sr_model, dataset_type='visdrone')
    # crop2sr('/data/VisDronePlus200', '/data/VisDronePP200', sr_model, dataset_type='visdrone', sr_size=300)