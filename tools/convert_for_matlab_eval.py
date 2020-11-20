import numpy as np
from tqdm import tqdm
import json
import os
from os import mknod, makedirs
from os.path import join, exists
import math


def convert_predictions(gt_json_file, pred_json_file, dst_dir):
    gt_json = json.loads(open(gt_json_file, 'r').read())
    pred_json = json.loads(open(pred_json_file, 'r').read())
    images = gt_json['images']
    pred_index = 0
    if not exists(dst_dir):
        makedirs(dst_dir)
    for image_i in tqdm(range(len(images))):
        image = images[image_i]
        image_id = image['id']
        txt_name = image['file_name'].split('.')[0] + '.txt'
        write_path = join(dst_dir, txt_name)
        if not exists(write_path):
            mknod(write_path)
        write_file = open(write_path, 'w')
        results_list = []
        while pred_index < len(pred_json) and pred_json[pred_index]['image_id'] == image_id:
            box = pred_json[pred_index]['bbox']
            score = pred_json[pred_index]['score']
            category = pred_json[pred_index]['category_id']
            msg = str(int(box[0])) + ',' + \
                  str(int(box[1])) + ',' + \
                  str(int(box[2])) + ',' + \
                  str(int(box[3])) + ',' + \
                  str(float(score)) + ',' + \
                  str(category) + ',-1,-1\n'
            results_list.append(msg)
            pred_index += 1
        if len(results_list)==0:
            print(image['file_name'])
        write_file.writelines(results_list)


if __name__ == "__main__":
    convert_predictions('./datasets/VisDrone2019-DET/annotations/val.json',
                        './train_log/1028_faster_rcnn_res50_visdronepp/inference/coco_instances_results.json',
                        './cocores/frcnn_res50_visdronepp')
