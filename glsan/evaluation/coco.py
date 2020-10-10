import json
import time

from pycocotools.coco import COCO as Base
from collections import defaultdict

from slender_det.structures.masks import PolygonMasks


class COCO(Base):
    oriented = True
    def __init__(self, annotation_file=None, oriented=True):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :param oriented (bool): Use oriented box for computing ratio or not.
        :return:
        """
        self.oriented = oriented
        super(COCO, self).__init__(annotation_file)

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list),defaultdict(list)

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                self.compute_ratio(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    @classmethod
    def compute_ratio(self, ann, oriented=None):
        if "ratio" in ann:
            return ann

        if oriented is None:
            oriented = self.oriented

        if "segmentation" in ann:
            segm = [poly for poly in ann["segmentation"] if len(poly) % 2 == 0 and len(poly) >= 6]
        else:
            segm = []
        if ann["iscrowd"] or\
                "segmentation" not in ann or\
                len(segm) == 0:
            w, h = ann["bbox"][2], ann["bbox"][3]
            if not oriented:
                ann["ratio"] = w / h
            else:
                ann["ratio"] = min(w, h) / max(w, h)
            return ann

        segmentations = PolygonMasks([segm])
        ratio = segmentations.get_ratios(oriented=oriented)[0]
        ann["ratio"] = ratio
        return ann
