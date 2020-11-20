# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn

from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from detectron2.data import detection_utils as d2utils
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from glsan.utils import uniformly_crop, self_adaptive_crop, cluster_by_boxes_centers
from .edsr import EDSR
import math

__all__ = ["GlsanNet"]


@META_ARCH_REGISTRY.register()
class GlsanNet(GeneralizedRCNN):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.image_format = cfg.INPUT.FORMAT
        self.crop_method = cfg.GLSAN.CROP
        self.cluster_num = cfg.GLSAN.CLUSTER_NUM
        self.crop_size = cfg.GLSAN.CROP_SIZE
        self.padding_size = cfg.GLSAN.PADDING_SIZE
        self.normalized_ratio = cfg.GLSAN.NORMALIZED_RATIO
        self.sr = cfg.GLSAN.SR
        self.sr_thresh = cfg.GLSAN.SR_THRESH
        self.sr_model = EDSR().to(self.device)
        self.sr_model.load_state_dict(torch.load('./models/visdrone_x2.pt'))

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        assert self.crop_method in ["NoCrop", "UniformlyCrop", "SelfAdaptiveCrop"], "crop method not in given range!"
        results = self.batched_inference(batched_inputs, detected_instances, do_postprocess)
        for r_i in range(len(results)):
            image = d2utils.read_image(batched_inputs[r_i]["file_name"], format=self.image_format)
            self.visualize_boxes(results[r_i], image.copy(),
                                 './visualization/baseline/' + batched_inputs[r_i]["file_name"].split('/')[-1],
                                 show_score=False, show_class=False)
        if self.crop_method == "NoCrop":
            return results
        images = []
        offsets = []
        part_imgs = []
        aug_images = []
        augs = d2utils.build_augmentation(self.cfg, self.training)
        part_results = []
        for i in range(len(batched_inputs)):
            image = d2utils.read_image(batched_inputs[i]["file_name"], format=self.image_format)
            images.append(image)
            if self.crop_method == "UniformlyCrop":
                offsets_per_img, part_imgs_per_img = uniformly_crop(image)
            elif self.crop_method == "SelfAdaptiveCrop":
                offsets_per_img, part_imgs_per_img = \
                    self_adaptive_crop(results[i]['instances'].pred_boxes.tensor.cpu().numpy(), image, self.cluster_num,
                                       self.crop_size, self.padding_size, self.normalized_ratio)
            offsets.append(offsets_per_img)
            part_imgs.append(part_imgs_per_img)

            aug_inputs_per_img = []
            for img_i in range(len(part_imgs_per_img)):
                image = part_imgs_per_img[img_i]
                image_shape = image.shape[0:2]
                if self.sr:
                    # super-resolution
                    image_size = math.sqrt(image_shape[0] * image_shape[1])
                    if image_size <= self.sr_thresh:
                        sr_input = torch.FloatTensor(image.copy()).to(self.device).permute(2, 0, 1).unsqueeze(0)
                        image = self.sr_model(sr_input)
                        image = image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                aug_input = T.StandardAugInput(image)
                aug_input.apply_augmentations(augs)
                aug_image = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(self.device)
                part_aug_input = dict()
                part_aug_input['image'] = aug_image
                part_aug_input['height'], part_aug_input['width'] = image_shape
                part_aug_input['file_name'] = batched_inputs[i]['file_name']
                part_aug_input['image_id'] = batched_inputs[i]['image_id']
                aug_inputs_per_img.append(part_aug_input)
            part_result = self.batched_inference(aug_inputs_per_img, detected_instances, do_postprocess)
            part_results.append(part_result)
            aug_images.append(aug_inputs_per_img)
        merged_results = self.merge_results(results, part_results, offsets, merge_mode='merge')
        for r_i in range(len(merged_results)):
            self.visualize_boxes(merged_results[r_i], images[r_i].copy(),
                                 './visualization/ori/' + batched_inputs[r_i]["file_name"].split('/')[-1],
                                 show_score=False, show_class=False)
        return merged_results

    def batched_inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        if len(batched_inputs) == 0:
            return []
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GlsanNet._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def merge_results(self, results, part_results, offsets, merge_mode='merge'):
        result_per_image = [
            self.merge_results_single_image(result, part_result, offset, merge_mode)
            for result, part_result, offset in zip(results, part_results, offsets)
        ]
        return result_per_image

    def merge_results_single_image(self, result, part_result, offset, merge_mode='merge'):
        assert merge_mode in ['global', 'local', 'merge'], 'merge mode must in [\'global\', \'local\', \'merge\']!'
        if len(part_result) == 0:
            return result
        result = result['instances']
        merged_result = Instances(result.image_size)
        if merge_mode == 'global':
            return {'instances': result}
        elif merge_mode == 'local':
            merged_boxes = []
            merged_scores = []
            merged_pred_classes = []
        else:
            merged_boxes = [result.pred_boxes.tensor]
            merged_scores = [result.scores]
            merged_pred_classes = [result.pred_classes]

        for i in range(len(part_result)):
            part_instance = part_result[i]['instances']
            part_boxes = part_instance.pred_boxes.tensor
            part_offset = torch.tensor(offset[i]).to(self.device).flip(0).repeat(part_boxes.shape[0], 2)

            merged_boxes.append(part_boxes + part_offset)
            merged_scores.append(part_instance.scores)
            merged_pred_classes.append(part_instance.pred_classes)

        merged_boxes = torch.cat(merged_boxes, dim=0)
        merged_scores = torch.cat(merged_scores, dim=0)
        merged_pred_classes = torch.cat(merged_pred_classes, dim=0)

        # Apply per-class NMS
        keep = batched_nms(merged_boxes, merged_scores, merged_pred_classes, self.test_nms_thresh)
        if self.test_topk_per_image >= 0:
            keep = keep[:self.test_topk_per_image]
        boxes, scores, pred_classes = merged_boxes[keep], merged_scores[keep], merged_pred_classes[keep]

        merged_result.pred_boxes = Boxes(boxes)
        merged_result.scores = scores
        merged_result.pred_classes = pred_classes
        return {'instances': merged_result}

    def visualize_boxes(self, result, image, file_name, show_score=False, show_class=False):
        img = Image.fromarray(image[...,::-1])
        draw = ImageDraw.Draw(img)
        pred_boxes = result['instances'].pred_boxes.tensor.cpu().numpy().astype(np.int32)
        scores = result['instances'].scores.cpu().numpy()
        pred_classes = result['instances'].pred_classes.cpu().numpy()
        meta = MetadataCatalog.get('visdrone_train')
        font = ImageFont.truetype('LiberationSans-Regular.ttf', 20)
        for box_i, pred_box in enumerate(pred_boxes):
            if scores[box_i] < 0.3: continue
            color = tuple(meta.thing_colors[pred_classes[box_i]])
            draw.rectangle([pred_box[0], pred_box[1], pred_box[2], pred_box[3]], outline=color)
            if show_score:
                score = scores[box_i]
                draw.text((pred_box[2], pred_box[1]),
                          str(np.around(score, decimals=2)), font=font, fill=color)
            if show_class:
                pred_class = meta.thing_classes[pred_classes[box_i]]
                draw.text((pred_box[2] + 40, pred_box[1]), pred_class, font=font, fill=color)
        img.save(file_name)
