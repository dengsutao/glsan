#!/usr/bin/python
# xml是voc的格式
# json是coco的格式
import sys, os, json, glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from os import mknod
from os.path import exists, join
INITIAL_BBOXIds = 1
# PREDEF_CLASSE = {}
PREDEF_CLASSE = {'car': 1, 'truck': 2, 'bus': 3}
val_folds = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801', 'M0802',
              'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']  # 20

# function
def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if 0 < length != len(vars):
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_path, out_json, split):
    if not exists(out_json):
        mknod(out_json)
    xml_paths = []
    for xml_seq_path in os.listdir(xml_path):
        if (xml_seq_path in val_folds and split == 'val') or (xml_seq_path not in val_folds and split == 'train'):
            xml_seq_dir = join(xml_path, xml_seq_path)
            for xml_file in os.listdir(xml_seq_dir):
                xml_paths.append(join(xml_seq_dir, xml_file))

    json_dict = {'images': [], 'type': 'instances',
                 'categories': [], 'annotations': []}
    categories = PREDEF_CLASSE
    bbox_id = INITIAL_BBOXIds
    for image_id, xml_f in tqdm(enumerate(xml_paths)):
        seq_name = xml_f.split('/')[-2]
        tree = ET.parse(xml_f)
        root = tree.getroot()
        filename = get_and_check(root, 'filename', 1).text
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': join(seq_name, filename), 'height': height,
                 'width': width, 'id': image_id + 1}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = max(categories.values()) + 1
                categories[category] = new_id
            category_id = categories[category]
            bbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bbox, 'xmin', 1).text)
            ymin = int(get_and_check(bbox, 'ymin', 1).text)
            xmax = int(get_and_check(bbox, 'xmax', 1).text)
            ymax = int(get_and_check(bbox, 'ymax', 1).text)
            if xmax <= xmin or ymax <= ymin:
                continue
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id + 1,
                   'bbox': [xmin, ymin, o_width, o_height], 'category_id': category_id,
                   'id': bbox_id, 'ignore': 0, 'segmentation': []}
            json_dict['annotations'].append(ann)
            bbox_id = bbox_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    # json_file = open(out_json, 'w')
    # json_str = json.dumps(json_dict)
    # json_file.write(json_str)
    # json_file.close() # 快
    json.dump(json_dict, open(out_json, 'w'), indent=4)  # indent=4 更加美观显示 慢


if __name__ == '__main__':
    split = 'val'
    xml_path = '/data/UAV-benchmark-MOTD_v1.0/Annotations_XML/'
    convert(xml_path, '/data/UAV-benchmark-MOTD_v1.0/annotations/val.json', split)
    # train 23829
    # val 16580
