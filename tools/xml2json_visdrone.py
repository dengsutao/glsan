#!/usr/bin/python
# xml是voc的格式
# json是coco的格式
import sys, os, json, glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from os import mknod
from os.path import exists
INITIAL_BBOXIds = 1
# PREDEF_CLASSE = {}
PREDEF_CLASSE = {'pedestrian': 1, 'people': 2,
                 'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6, 'tricycle': 7,
                 'awning-tricycle': 8, 'bus': 9, 'motor': 10}



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


def convert(xml_paths, out_json):
    if not exists(out_json):
        mknod(out_json)
    json_dict = {'images': [], 'type': 'instances',
                 'categories': [], 'annotations': []}
    categories = PREDEF_CLASSE
    bbox_id = INITIAL_BBOXIds
    for image_id, xml_f in tqdm(enumerate(xml_paths)):


        tree = ET.parse(xml_f)
        root = tree.getroot()
        filename = get_and_check(root, 'filename', 1).text
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height,
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
    xml_path = '/data/VisDrone2019-DET-val_dense/Annotations_XML/'
    xml_file = glob.glob(os.path.join(xml_path, '*.xml'))
    convert(xml_file, '/data/VisDrone2019-DET-val_dense/annotations/val.json')
