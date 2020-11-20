import os
from PIL import Image
from tqdm import tqdm
from os.path import exists, join
from os import mknod, makedirs

# 把下面的路径改成你自己的路径即可
# root_dir = "/data/VisDrone2019-DET-val/"
root_dir = "/data/UAV-benchmark-MOTD_v1.0/"
annotations_dir = root_dir + "GT/"
image_dir = "/data/UAV-benchmark-M/"
xml_dir = root_dir + "Annotations_XML/"
if not exists(xml_dir):
    makedirs(xml_dir)
# 下面的类别也换成你自己数据类别，也可适用于其他的数据集转换
class_name = ['car', 'truck', 'bus']

uavdt_objects = dict()
for filename in tqdm(os.listdir(annotations_dir)):
    if 'whole' not in filename:  # for DET
        continue

    seq_name = filename.split('_')[0]
    with open(annotations_dir + filename, 'r') as f:
        lines = f.readlines()

    single_seq_objects = dict()
    for line in lines:
        image_idx, object_id, xmin, ymin, w, h, oov, occlusion, category = [int(_) for _ in line.rstrip().split(',')]
        xmax = xmin + w
        ymax = ymin + h
        if "img{:0>6d}".format(image_idx) not in single_seq_objects:
            single_seq_objects["img{:0>6d}".format(image_idx)] = [[xmin, ymin, xmax, ymax, category]]
        else:
            single_seq_objects["img{:0>6d}".format(image_idx)].append([xmin, ymin, xmax, ymax, category])
    uavdt_objects[seq_name] = single_seq_objects

for seq_name, single_seq_objects in tqdm(uavdt_objects.items()):
    xml_seq_dir = join(xml_dir, seq_name)
    if not exists(xml_seq_dir):
        makedirs(xml_seq_dir)
    for img_name, single_img_objects in single_seq_objects.items():
        img = Image.open(join(image_dir, seq_name, img_name + '.jpg'))
        xml_name = join(xml_seq_dir, img_name + '.xml')
        if not exists(xml_name):
            mknod(xml_name)
        with open(xml_name, 'w') as fout:
            fout.write('<annotation>' + '\n')

            fout.write('\t' + '<folder>VOC2007</folder>' + '\n')
            fout.write('\t' + '<filename>' + img_name + '.jpg' + '</filename>' + '\n')

            fout.write('\t' + '<source>' + '\n')
            fout.write('\t\t' + '<database>' + 'UAV-benchmark-M' + '</database>' + '\n')
            fout.write('\t\t' + '<annotation>' + 'UAV-benchmark-M' + '</annotation>' + '\n')
            fout.write('\t\t' + '<image>' + 'flickr' + '</image>' + '\n')
            fout.write('\t\t' + '<flickrid>' + 'Unspecified' + '</flickrid>' + '\n')
            fout.write('\t' + '</source>' + '\n')

            fout.write('\t' + '<size>' + '\n')
            fout.write('\t\t' + '<width>' + str(img.size[0]) + '</width>' + '\n')
            fout.write('\t\t' + '<height>' + str(img.size[1]) + '</height>' + '\n')
            fout.write('\t\t' + '<depth>' + '3' + '</depth>' + '\n')
            fout.write('\t' + '</size>' + '\n')

            fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')

            for bbox in single_img_objects:
                if not (0 < bbox[4] < 4):
                    continue
                fout.write('\t' + '<object>' + '\n')
                fout.write('\t\t' + '<name>' + class_name[bbox[4]-1] + '</name>' + '\n')
                fout.write('\t\t' + '<bndbox>' + '\n')
                fout.write('\t\t\t' + '<xmin>' + str(bbox[0]) + '</xmin>' + '\n')
                fout.write('\t\t\t' + '<ymin>' + str(bbox[1]) + '</ymin>' + '\n')
                fout.write('\t\t\t' + '<xmax>' + str(bbox[2]) + '</xmax>' + '\n')
                fout.write('\t\t\t' + '<ymax>' + str(bbox[3]) + '</ymax>' + '\n')
                fout.write('\t\t' + '</bndbox>' + '\n')
                fout.write('\t' + '</object>' + '\n')

            fout.write('</annotation>')
