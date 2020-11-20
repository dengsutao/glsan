import math
import json
from tqdm import tqdm

visdrone_categories = {'pedestrian': 1, 'people': 2,
                       'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6, 'tricycle': 7,
                       'awning-tricycle': 8, 'bus': 9, 'motor': 10}
uavdt_categories = {'car': 1, 'truck': 2, 'bus': 3}


def object_statistic(json_file, categories=visdrone_categories):
    json_dict = json.loads(open(json_file, 'r').read())
    annos = json_dict['annotations']
    avg_box_size = 0
    total_box_num = 0
    avg_category_box_size = [0 for x in categories.items()]
    category_num = [0 for x in categories.items()]
    category_rate = [0 for x in categories.items()]
    max_sizes = [0 for x in categories.items()]
    min_sizes = [100000 for x in categories.items()]
    for i in tqdm(range(len(annos))):
        anno = annos[i]
        bbox = anno['bbox']
        area = anno['area']
        bbox_size = math.sqrt(area)
        category_id = anno['category_id'] - 1
        category_num[category_id] += 1
        total_box_num += 1
        avg_category_box_size[category_id] += bbox_size
        if bbox_size > max_sizes[category_id]:
            max_sizes[category_id] = bbox_size
        if bbox_size < min_sizes[category_id]:
            min_sizes[category_id] = bbox_size
    for i in range(len(category_num)):
        avg_category_box_size[i] = avg_category_box_size[i] / category_num[i]
        category_rate[i] = category_num[i] / total_box_num
    print(categories)
    print(avg_category_box_size)
    print(category_num)
    print(category_rate)
    print(max_sizes)
    print(min_sizes)


def image_statistic(json_file):
    json_dict = json.loads(open(json_file, 'r').read())
    annos = json_dict['annotations']
    image_infos = json_dict['images']
    total_nums = 0
    size_nums = [0 for x in range(15)]
    size_rates = [0 for x in range(15)]
    max_size = 0
    min_size = 100000
    for i in tqdm(range(len(image_infos))):
        image_info = image_infos[i]
        image_name = image_info['file_name']
        if len(image_name.split('_')[-1].split('.')[0]) == 1:
            h = image_info['height']
            w = image_info['width']
            total_nums += 1
            image_size = int(math.sqrt(h * w))
            if image_size > 1300:
                print(image_name)
            size_nums[image_size // 100] += 1
            if max_size < image_size:
                max_size = image_size
            if min_size > image_size:
                min_size = image_size
    for i in range(len(size_nums)):
        size_rates[i] = round(size_nums[i] / total_nums, 2)
    print(size_nums)
    print(size_rates)


if __name__ == '__main__':
    object_statistic('/data/VisDrone2019-DET/annotations/val.json')
    # image_statistic('/data/VisDronePlus200/annotations/train.json')
