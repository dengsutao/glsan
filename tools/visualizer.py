from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visualize_gt_boxes(boxes, classes, image, file_name,
                       meta_name='visdrone_train', show_class=False):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    meta = MetadataCatalog.get(meta_name)
    font = ImageFont.truetype('LiberationSans-Regular.ttf', 20)
    for box_i, box in enumerate(boxes):
        color = tuple(meta.thing_colors[classes[box_i]-1])
        draw.rectangle(box, outline=color)
        if show_class:
            thing_class = meta.thing_classes[classes[box_i]-1]
            draw.text((box[2] + 40, box[1]), thing_class, font=font, fill=color)
    img.save(file_name)
