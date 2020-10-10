from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_backbone, Backbone
from . import ContrastRPN
from .utils import INF, permute_predicts, get_num_gpus, reduce_sum
import math
import cv2
import numpy as np
import os
from os import path


def visualize_instance_mask(mask, points_in_box, points_in_mask):
    for i in range(len(mask)):
        img = (mask[i] * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dict = {}
        for pts in points_in_mask[i]:
            coor = tuple(pts.tolist())
            if (not coor in dict) or dict[coor] == 0:
                dict[coor] = 1
                cv2.circle(img, coor, 3, (0, 0, 255), -1)
        for pts in points_in_box[i]:
            coor = tuple(pts.tolist())
            if not coor in dict:
                dict[coor] = 0
                cv2.circle(img, coor, 3, (0, 255, 0), -1)
        if not path.exists('./visualization'):
            os.mkdir('./visualization')
        cv2.imwrite('./visualization/instance_mask_' + str(i) + '.jpg', img)


def visualize_image_mask(mask, points_in_box, points_in_mask):
    img = (mask * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dict = {}  # [values 1: in mask, 0: in box]
    for level_pts in points_in_mask:
        for pts in level_pts:
            coor = tuple(pts.tolist())
            if (not coor in dict) or dict[coor] == 0:
                dict[coor] = 1
                cv2.circle(img, coor, 3, (0, 0, 255), -1)
    for level_pts in points_in_box:
        for pts in level_pts:
            coor = tuple(pts.tolist())
            if not coor in dict:
                dict[coor] = 0
                cv2.circle(img, coor, 3, (0, 255, 0), -1)
    if not path.exists('./visualization'):
        os.mkdir('./visualization')
    cv2.imwrite('./visualization/image_mask.jpg', img)


def compute_targets_for_locations(
        locations, targets, gt_center_masks, gt_instance_masks,
        gt_instance_mask_offsets, neg_points_thresh, num_classes, strides
):
    NUM_SAMPLE_POINTS = 30
    masks_level_all = []
    inds_in_dilated_mask_all = []
    inds_in_mask_all = []
    centerness_in_mask_all = []
    for im_i in range(len(targets)):
        gt_instance_mask = gt_instance_masks[im_i]
        gt_instance_mask_offset = gt_instance_mask_offsets[im_i]

        masks_level = []
        points_in_box = []
        points_in_dilated_mask = []
        points_in_mask = []
        inds_in_box = []
        inds_in_dilated_mask = []
        inds_in_mask = []
        centerness_in_mask = []

        points_in_box_wo_shift = []
        points_in_dilated_mask_wo_shift = []
        points_in_mask_wo_shift = []
        for mask_i in range(len(gt_instance_mask)):
            gt_instance_mask_i = torch.from_numpy(gt_instance_mask[mask_i]).to(locations[0].device)
            gt_instance_mask_offset_i = torch.from_numpy(gt_instance_mask_offset[mask_i]).to(locations[0].device)
            mask_h, mask_w = gt_instance_mask_i.shape
            shift_w, shift_h = gt_instance_mask_offset_i
            mask_size = math.sqrt(mask_h * mask_w)
            mask_box = torch.Tensor([shift_w, shift_h, shift_w + mask_w, shift_h + mask_h]).long().to(
                locations[0].device)
            for level in range(len(locations) - 1, -1, -1):
                location = locations[level].long()
                xs, ys = location[:, 0], location[:, 1]
                l = xs - mask_box[0]
                t = ys - mask_box[1]
                r = mask_box[2] - xs
                b = mask_box[3] - ys
                ltrb_offset = torch.stack([l, t, r, b], dim=1)
                is_in_box = ltrb_offset.min(dim=1)[0] > 0
                in_box_inds = is_in_box.nonzero().view(-1)
                in_box_location = location[in_box_inds]
                in_box_location_wo_shift = in_box_location - gt_instance_mask_offset_i.repeat(
                    len(in_box_location), 1)

                is_in_dilated_mask = (gt_instance_mask_i[in_box_location_wo_shift[:, 1],
                                                         in_box_location_wo_shift[:, 0]] <= neg_points_thresh) & \
                                     (gt_instance_mask_i[in_box_location_wo_shift[:, 1],
                                                         in_box_location_wo_shift[:, 0]] > 0)
                in_dilated_mask_inds = in_box_inds[is_in_dilated_mask.nonzero().view(-1)]
                in_dilated_mask_location = location[in_dilated_mask_inds]
                in_dilated_mask_location_wo_shift = in_dilated_mask_location - gt_instance_mask_offset_i.repeat(
                    len(in_dilated_mask_location), 1)

                is_in_mask = gt_instance_mask_i[in_box_location_wo_shift[:, 1],
                                                in_box_location_wo_shift[:, 0]] > neg_points_thresh + 1e-5
                in_mask_inds = in_box_inds[is_in_mask.nonzero().view(-1)]
                in_mask_location = location[in_mask_inds]
                in_mask_location_wo_shift = in_mask_location - gt_instance_mask_offset_i.repeat(
                    len(in_mask_location), 1)

                in_mask_centerness = gt_instance_mask_i[in_mask_location_wo_shift[:, 1],
                                                        in_mask_location_wo_shift[:, 0]]

                if len(in_mask_inds) >= NUM_SAMPLE_POINTS or level == 0:
                    centerness_in_mask.append(in_mask_centerness)
                    masks_level.append(level)
                    points_in_box.append(in_box_location)
                    inds_in_box.append(in_box_inds)
                    points_in_dilated_mask.append(in_dilated_mask_location)
                    inds_in_dilated_mask.append(in_dilated_mask_inds)
                    points_in_mask.append(in_mask_location)
                    inds_in_mask.append(in_mask_inds)
                    points_in_box_wo_shift.append(in_box_location_wo_shift)
                    points_in_dilated_mask_wo_shift.append(in_dilated_mask_location_wo_shift)
                    points_in_mask_wo_shift.append(in_mask_location_wo_shift)
                    break
        masks_level = torch.tensor(masks_level).to(locations[0].device)

        visualize_instance_mask(gt_instance_mask, points_in_dilated_mask_wo_shift, points_in_mask_wo_shift)
        visualize_image_mask(gt_center_masks[im_i], points_in_dilated_mask, points_in_mask)

        masks_level_all.append(masks_level)
        inds_in_dilated_mask_all.append(inds_in_dilated_mask)
        inds_in_mask_all.append(inds_in_mask)
        centerness_in_mask_all.append(centerness_in_mask)

    return masks_level_all, inds_in_dilated_mask_all, inds_in_mask_all, centerness_in_mask_all


@META_ARCH_REGISTRY.register()
class PSONet(ContrastRPN):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.neg_points_thresh = 0.001

    @torch.no_grad()
    def get_ground_truth(self, points, gt_instances):
        standard = "linear"
        sigma = 0.5
        gt_center_masks = []
        gt_instance_masks = []
        gt_instance_mask_offsets = []
        # compute center_score
        for im_i in range(len(gt_instances)):
            gt_per_im = gt_instances[im_i]
            gt_masks = gt_per_im.gt_masks
            # standard = 'linear' or 'gaussian', sigma used for gaussian distribution
            center_masks_i, instance_center_mask_i, instance_mask_offset_i = \
                gt_masks.center_masks(mask_size=gt_per_im.image_size, standard=standard,
                                      sigma=sigma, dilate_rate=2, neg_points_thresh=self.neg_points_thresh)

            gt_center_masks.append(center_masks_i)
            gt_instance_masks.append(instance_center_mask_i)
            gt_instance_mask_offsets.append(instance_mask_offset_i)

        # cv2.imwrite('mask.jpg', (gt_center_masks[0] * 255).astype(np.uint8))

        masks_level_all, inds_in_dilated_mask_all, inds_in_mask_all, centerness_in_mask_all = \
            compute_targets_for_locations(
                points, gt_instances,
                gt_center_masks, gt_instance_masks, gt_instance_mask_offsets,
                self.neg_points_thresh, self.num_classes, self.fpn_stride
            )
        return masks_level_all, inds_in_dilated_mask_all, inds_in_mask_all, centerness_in_mask_all

    def losses(self, masks_level_all, inds_in_dilated_mask_all,
               inds_in_mask_all, centerness_in_mask_all, cls_outs, embed_outs):
        # list([N,H*W,C])
        cls_outs, embed_outs = permute_predicts(cls_outs, embed_outs, self.head.embed_channels)
        pred_centerness = []
        gt_centerness = []
        pred_cos_distance = []
        gt_cos_distance = []
        for level_i in range(len(cls_outs)):
            pred_cls = cls_outs[level_i]
            pred_features = embed_outs[level_i]

            for im_i in range(len(masks_level_all)):
                masks_level = masks_level_all[im_i]
                inds_in_dilated_masks = inds_in_dilated_mask_all[im_i]
                inds_in_masks = inds_in_mask_all[im_i]
                centerness_in_masks = centerness_in_mask_all[im_i]
                for mask_i in range(len(masks_level)):
                    inds_in_dilated_mask = inds_in_dilated_masks[mask_i]
                    inds_in_mask = inds_in_masks[mask_i]
                    centerness_in_mask = centerness_in_masks[mask_i]

                    if not masks_level[mask_i] == level_i or len(inds_in_mask) == 0:
                        continue
                    # centerness loss
                    gt_centerness.append(centerness_in_mask)
                    gt_cos_distance_i = torch.cat((torch.ones((len(inds_in_mask))),
                                                   torch.zeros((len(inds_in_dilated_mask)))), dim=0).to(self.device)
                    pred_centerness.append(pred_cls[im_i, inds_in_mask, :])
                    sorted_centerness, inds = torch.topk(centerness_in_mask, 1)
                    max_centerness_inds = inds[0]
                    max_centerness = sorted_centerness[0]
                    max_centerness_feature = pred_features[im_i, max_centerness_inds, :]
                    pred_features_in_mask = pred_features[im_i, inds_in_mask, :]
                    pred_features_in_dilated_mask = pred_features[im_i, inds_in_dilated_mask, :]
                    cos_distance_inside = torch.cosine_similarity(max_centerness_feature.repeat(len(inds_in_mask), 1),
                                                                  pred_features_in_mask, dim=1)
                    cos_distance_outside = torch.cosine_similarity(
                        max_centerness_feature.repeat(len(inds_in_dilated_mask), 1),
                        pred_features_in_dilated_mask, dim=1)
                    pred_cos_distance_i = torch.cat((cos_distance_inside, cos_distance_outside), dim=0)
                    gt_cos_distance.append(gt_cos_distance_i)
                    pred_cos_distance.append(pred_cos_distance_i)
        num_gpus = get_num_gpus()
        pred_centerness = torch.cat(pred_centerness).view(-1)
        gt_centerness = torch.cat(gt_centerness)
        total_num_pos_ctn = reduce_sum(torch.tensor([pred_centerness.numel()]).to(self.device)).item()
        num_pos_ctn_avg_per_gpu = max(total_num_pos_ctn / float(num_gpus), 1.0)

        centerness_loss = F.binary_cross_entropy_with_logits(
            pred_centerness, gt_centerness, reduction='sum'
        ) / num_pos_ctn_avg_per_gpu

        pred_cos_distance = torch.cat(pred_cos_distance)
        gt_cos_distance = torch.cat(gt_cos_distance)
        total_num_pos_dist = reduce_sum(torch.tensor([pred_cos_distance.numel()]).to(self.device)).item()
        num_pos_dist_avg_per_gpu = max(total_num_pos_dist / float(num_gpus), 1.0)

        distance_loss = F.binary_cross_entropy_with_logits(
            pred_cos_distance, gt_cos_distance, reduction='sum'
        ) / num_pos_dist_avg_per_gpu

        return dict(centerness_loss=centerness_loss, distance_loss=distance_loss)
