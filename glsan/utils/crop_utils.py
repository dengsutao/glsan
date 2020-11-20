import numpy as np
import time
from sklearn.cluster import KMeans
import math

__all__ = ['uniformly_crop', 'self_adaptive_crop', 'cluster_by_boxes_centers']


def uniformly_crop(img_ori):
    height_ori = img_ori.shape[0]
    width_ori = img_ori.shape[1]
    offsets = []
    imgs = []
    offsets.append([0, 0])
    imgs.append(img_ori[0:height_ori // 2, 0:width_ori // 2, :])

    offsets.append([0, width_ori // 2])
    imgs.append(img_ori[0:height_ori // 2, width_ori // 2:, :])

    offsets.append([height_ori // 2, 0])
    imgs.append(img_ori[height_ori // 2:, 0:width_ori // 2, :])

    offsets.append([height_ori // 2, width_ori // 2])
    imgs.append(img_ori[height_ori // 2:, width_ori // 2:, :])

    return offsets, imgs


def self_adaptive_crop(boxes, img_ori, cluster_num=4, crop_size=300, padding_size=50, normalized_ratio=2):
    height_ori = img_ori.shape[0]
    width_ori = img_ori.shape[1]
    center_map = np.zeros((height_ori, width_ori))
    centers, ranges = cluster_by_boxes_centers(cluster_num, center_map, boxes,
                                               crop_size, padding_size, normalized_ratio)
    imgs = []
    offsets = []
    for i, center in enumerate(centers):
        range = ranges[i]
        # part_x1, part_y1, part_x2, part_y2 = clamp_range(center, range, height_ori, width_ori)
        part_x1, part_y1, part_x2, part_y2 = transfrom_offsets(center, range, height_ori, width_ori)
        part_img = img_ori[part_y1:part_y2, part_x1:part_x2, :]
        offsets.append([part_y1, part_x1])
        imgs.append(part_img)
    return offsets, imgs


def clamp_range(center, range, height_ori, width_ori):
    # [x,y,x,y]
    r = [max(0, center[1] - range[1] // 2),
         max(0, center[0] - range[0] // 2),
         min(width_ori, center[1] + range[1] // 2),
         min(height_ori, center[0] + range[0] // 2)]
    return r


def transfrom_offsets(center, range, height_ori, width_ori):
    center_x = center[1]
    center_y = center[0]
    crop_width = range[1]
    crop_height = range[0]
    part_x1 = int(center_x - crop_width // 2)
    part_x2 = int(center_x + crop_width // 2)
    part_y1 = int(center_y - crop_height // 2)
    part_y2 = int(center_y + crop_height // 2)
    if part_x1 < 0 and part_x2 > width_ori:
        center_x = int(width_ori // 2)
        part_x1 = 0
        part_x2 = width_ori
    elif part_x1 < 0:
        offset_x = 0 - part_x1
        center_x += offset_x
        part_x1 += offset_x
        part_x2 += offset_x
        if part_x2 > width_ori:
            center_x += (width_ori - part_x2) / 2
            part_x2 = width_ori
    elif part_x2 > width_ori:
        offset_x = width_ori - part_x2
        center_x += offset_x
        part_x1 += offset_x
        part_x2 += offset_x
        if part_x1 < 0:
            center_x += (0 - part_x1) / 2
            part_x1 = 0
    if part_y1 < 0 and part_y2 > height_ori:
        center_y = int(height_ori // 2)
        part_y1 = 0
        part_y2 = height_ori
    elif part_y1 < 0:
        offset_y = 0 - part_y1
        center_y += offset_y
        part_y1 += offset_y
        part_y2 += offset_y
        if part_y2 > height_ori:
            center_y += (height_ori - part_y2) / 2
            part_y2 = height_ori
    elif part_y2 > height_ori:
        offset_y = height_ori - part_y2
        center_y += offset_y
        part_y1 += offset_y
        part_y2 += offset_y
        if part_y1 < 0:
            center_y += (0 - part_y1) / 2
            part_y1 = 0
    return part_x1, part_y1, part_x2, part_y2


def cluster_by_boxes_centers(cluster_num, center_map, boxes,
                             crop_size=300, padding_size=50, normalized_ratio=2, weight_with_area=False):
    start = time.time()
    center2boxes = {}
    X = []
    weighted_X = []
    for p in boxes:
        x = int(p[0] + p[2]) // 2
        y = int(p[1] + p[3]) // 2
        if [y, x] not in X:
            X.append([y, x])
        weight = int((p[2] - p[0]) * (p[3] - p[1])) // 400 + 1
        if weight_with_area:
            for w in range(weight):
                weighted_X.append([y, x])
        center2boxes[(y, x)] = p
    if len(X) == 0:
        return [], []
    ranges = [[] for i in range(cluster_num)]
    centers = [[] for i in range(cluster_num)]

    if len(X) < cluster_num:
        for i in range(cluster_num):
            if i < len(X):
                centers[i] = X[i]
            else:
                centers[i] = X[0]

            if crop_size <= 0:
                ranges[i] = [300, 300]
            else:
                ranges[i] = [crop_size, crop_size]
        return np.asarray(centers), np.asarray(ranges)
    if weight_with_area:
        X = np.asarray(weighted_X)
    else:
        X = np.asarray(X)
    kmeans = KMeans(n_clusters=cluster_num)
    classes = kmeans.fit(X)
    end = time.time()
    cost_time = end - start

    lbs = classes.labels_
    for i in range(cluster_num):
        inds = np.where(lbs == i)
        tmp_h = X[inds[0]][:, 0]
        tmp_w = X[inds[0]][:, 1]
        assert len(tmp_h) > 0, "X len: {},inds len: {}".format(len(X), len(inds[0]))
        list_h = []
        list_w = []
        for j, h in enumerate(tmp_h):
            w = tmp_w[j]
            box = center2boxes[(h, w)]
            list_w.append(box[0])
            list_w.append(box[2])
            list_h.append(box[1])
            list_h.append(box[3])
        min_h = min(list_h)
        max_h = max(list_h)
        min_w = min(list_w)
        max_w = max(list_w)
        max_height = max_h - min_h + padding_size
        max_width = max_w - min_w + padding_size
        # ranges[i].append(max([max_height, max_width // normalized_ratio]))
        # ranges[i].append(max([max_height // normalized_ratio, max_width]))
        ranges[i].append(max([max_height, max_width // normalized_ratio, crop_size]))
        ranges[i].append(max([max_height // normalized_ratio, max_width, crop_size]))
        centers[i] = [(min_h + max_h) // 2, (min_w + max_w) // 2]
    ranges = np.asarray(ranges).astype(np.int32)
    centers = np.asarray(centers).astype(np.int32)

    return centers, ranges


def cluster_by_boxes_scatters(cluster_num, score_map, boxes, crop_size=300, padding_size=50, normalized_ratio=2):
    start = time.time()
    scatters = []
    boxes = boxes.astype(np.int32)
    for p in boxes:
        xs = range(p[0], p[2], 10)
        ys = range(p[1], p[3], 10)
        for x in xs:
            for y in ys:
                scatters.append([y, x])
    if len(scatters) == 0:
        return [], []

    scatters = np.asarray(scatters)
    ranges = [[] for i in range(cluster_num)]
    centers = [[] for i in range(cluster_num)]

    if len(scatters) < cluster_num:
        for i in range(cluster_num):
            if i < len(scatters):
                centers[i] = scatters[i]
                ranges[i] = [crop_size, crop_size]
            else:
                centers[i] = scatters[0]
                ranges[i] = [crop_size, crop_size]
        return np.asarray(centers), np.asarray(ranges)

    kmeans = KMeans(n_clusters=cluster_num)
    classes = kmeans.fit(scatters)
    end = time.time()
    cost_time = end - start

    lbs = classes.labels_
    for i in range(cluster_num):
        inds = np.where(lbs == i)
        list_h = scatters[inds[0]][:, 0]
        list_w = scatters[inds[0]][:, 1]
        min_h = min(list_h)
        max_h = max(list_h)
        min_w = min(list_w)
        max_w = max(list_w)
        max_height = max_h - min_h + padding_size
        max_width = max_w - min_w + padding_size
        # ranges[i].append(max([max_height, max_width // normalized_ratio]))
        # ranges[i].append(max([max_height // normalized_ratio, max_width]))
        ranges[i].append(max([max_height, max_width // normalized_ratio, crop_size]))
        ranges[i].append(max([max_height // normalized_ratio, max_width, crop_size]))
        centers[i] = [(min_h + max_h) // 2, (min_w + max_w) // 2]
    ranges = np.asarray(ranges).astype(np.int32)
    centers = np.asarray(centers).astype(np.int32)

    return centers, ranges
