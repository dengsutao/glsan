from typing import Any, Iterator, List, Union
from functools import lru_cache
import pyclipper

import numpy as np
import cv2

from detectron2.structures.masks import PolygonMasks, polygon_area
from detectron2.structures.boxes import Boxes

from concern.support import make_dual
from concern import webcv2

import torch
import math


@lru_cache()
def standard_linear(resolution=128, reverse=False, sigma=0, neg_points_thresh=0):
    grid = (np.mgrid[0:resolution, 0:resolution] / resolution).astype(np.float32).sum(0)
    if reverse:
        return (grid < 1) * (1 - grid) + (grid >= 1) * neg_points_thresh
    return (grid < 1) * grid + (grid >= 1) * neg_points_thresh


@lru_cache()
def standard_gaussian(resolution=128, reverse=False, sigma=0.5, neg_points_thresh=0):
    grid = (np.mgrid[0:resolution, 0:resolution] / resolution).astype(np.float32).sum(0)
    if reverse:
        grid = normal_distribution(grid, 0, sigma)
        grid = (grid >= grid[-1, 0]) * grid
        norm_grid = grid / grid[0, 0]
    else:
        grid = (grid < 1) * grid
        grid = normal_distribution(grid, 1, sigma)
        grid = (grid > grid[0, 0]) * grid
        norm_grid = grid / grid[-1, 0]
    return norm_grid


def triangle_padding(triangle, dilate_rate, neg_points_thresh):
    resolution = triangle.shape[0]
    new_resolution = int(resolution * math.sqrt(dilate_rate))
    grid = (np.mgrid[0:new_resolution, 0:new_resolution] / new_resolution).astype(np.float32).sum(0)
    new_matrix = np.ones((new_resolution, new_resolution)) * neg_points_thresh
    new_matrix[0:resolution, 0:resolution] = triangle
    new_matrix[grid >= 1] = 0

    return new_matrix


def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def coordinate_transform(standard: np.ndarray, p_o, p_x, p_y, out_shape):
    h, w = standard.shape[:2]
    source_points = np.array([[0, 0], [0, h], [w, 0]], dtype=np.float32)
    dest_points = np.array([p_o, p_y, p_x], dtype=np.float32)
    M = cv2.getAffineTransform(source_points, dest_points)
    return cv2.warpAffine(standard, M, out_shape)


def mask_in_triangle_with_padding(
        hull,
        width,
        height,
        point_o,
        mask,
        reverse=False,
        standard="linear",
        sigma=0.5,
        dilate_rate=1,
        neg_points_thresh=0
):
    assert standard in ["linear", "gaussian"], "standard must be linear or gaussian"
    assert dilate_rate >= 1, 'dilate_rate must >= 1!'
    if standard == "linear":
        standard = standard_linear
    else:
        standard = standard_gaussian
    padded_length_rate_all = math.sqrt(dilate_rate)
    point_x = hull[0]
    for next_i in range(1, hull.shape[0]):
        point_y = hull[next_i]
        local = coordinate_transform(
            triangle_padding(standard(reverse=reverse, sigma=sigma, neg_points_thresh=neg_points_thresh),
                             dilate_rate, neg_points_thresh),
            point_o, point_o + (point_x - point_o) * padded_length_rate_all,
                     point_o + (point_y - point_o) * padded_length_rate_all,
            (width, height))
        mask = np.maximum(mask, local)

        point_x = point_y

    point_y = hull[0]
    local = coordinate_transform(
        triangle_padding(standard(reverse=reverse, sigma=sigma, neg_points_thresh=neg_points_thresh),
                         dilate_rate, neg_points_thresh),
        point_o, point_o + (point_x - point_o) * padded_length_rate_all,
                 point_o + (point_y - point_o) * padded_length_rate_all,
        (width, height))
    mask = np.maximum(mask, np.clip(local, 0, 1))
    return mask


def mask_in_triangle(
        hull,
        width,
        height,
        point_o,
        mask,
        reverse=False,
        standard="linear",
        sigma=0.5
):
    assert standard in ["linear", "gaussian"], "standard must be linear or gaussian"
    if standard == "linear":
        standard = standard_linear
    else:
        standard = standard_gaussian
    point_x = hull[0]
    for next_i in range(1, hull.shape[0]):
        point_y = hull[next_i]
        local = coordinate_transform(
            # standard_linear(reverse=reverse),
            standard(reverse=reverse, sigma=sigma),
            point_o, point_x, point_y,
            (width, height))
        mask = np.maximum(mask, local)
        point_x = point_y

    point_y = hull[0]
    local = coordinate_transform(
        # standard_linear(reverse=reverse),
        standard(reverse=reverse, sigma=sigma),
        point_o, point_x, point_y,
        (width, height))
    mask = np.maximum(mask, np.clip(local, 0, 1))
    return mask


def dilate_polygon_by_rate(hull, dilate_rate):
    center = hull.mean(axis=0)
    dilate_hull = (center + (hull - center) * math.sqrt(dilate_rate)).astype(np.int32)
    return dilate_hull


def distance_in_triangle(
        hull,
        point_o,
        mask
):
    point_x = hull[0]
    for next_i in range(1, hull.shape[0]):
        point_y = hull[next_i]
        mask_canvas = mask.copy()
        distance = np.abs((point_x + point_y) / 2 - point_o)
        cv2.fillPoly(
            mask_canvas,
            [(np.array([point_x, point_y, point_o]) + 0.5).astype(np.int32).reshape(-1, 1, 2)],
            distance)
        mask = np.maximum(mask, mask_canvas)
        point_x = point_y

    point_y = hull[0]
    mask_canvas = mask.copy()
    distance = np.sqrt(np.square((point_x + point_y) / 2 - point_o).sum())
    cv2.fillPoly(
        mask_canvas,
        [(np.array([point_x, point_y, point_o]) + 0.5).astype(np.int32).reshape(-1, 1, 2)],
        distance)
    mask = np.maximum(mask, mask_canvas)
    return mask


def dilate_polygon(polygon: np.ndarray, distance):
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
    return np.array(padding.Execute(distance)[0])


def draw_border_map(polygon, canvas, ratio):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    distance = np.sqrt(polygon_area(polygon[:, 0], polygon[:, 1])) * ratio
    padded_polygon = dilate_polygon(polygon, distance)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros(
        (polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = distance_map.min(axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymax + height,
            xmin_valid - xmin:xmax_valid - xmax + width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


def compute_distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(
        xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(
        xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(
        point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 *
                     square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(
        square_distance_1, square_distance_2))[cosin < 0]
    # self.extend_line(point_1, point_2, result)
    return result


class BorderMasks(PolygonMasks):
    """
    This class stores borders of for all objects in one image, forming border graduating maps.

    Attributes:
        polygons: list[list[ndarray]]. Each ndarray is a float64 vector representing a polygon.
    """

    def __getitem__(self, item: Union[int, slice, List[int], torch.BoolTensor]) -> "PolygonMasks":
        """
        Support indexing over the instances and return a `PolygonMasks` object.
        `item` can be:

        1. An integer. It will return an object with only one instance.
        2. A slice. It will return an object with the selected instances.
        3. A list[int]. It will return an object with the selected instances,
           correpsonding to the indices in the list.
        4. A vector mask of type BoolTensor, whose length is num_instances.
           It will return an object with the instances whose mask is nonzero.
        """
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        elif isinstance(item, list):
            selected_polygons = [self.polygons[i] for i in item]
        elif isinstance(item, torch.Tensor):
            # Polygons is a list, so we have to move the indices back to CPU.
            if item.dtype == torch.bool:
                assert item.dim() == 1, item.shape
                item = item.nonzero().squeeze(1).cpu().numpy().tolist()
            elif item.dtype in [torch.int32, torch.int64]:
                item = item.cpu().numpy().tolist()
            else:
                raise ValueError("Unsupported tensor dtype={} for indexing!".format(item.dtype))
            selected_polygons = [self.polygons[i] for i in item]
        return BorderMasks(selected_polygons)

    def center_masks(self, mask_size, standard, sigma=0.5, dilate_rate=1, neg_points_thresh=0):
        center_mask = np.zeros(mask_size, dtype=np.float32)
        expansion_ratio = 0.1
        instances_center_mask = []
        instance_mask_offset = []
        for polygons_per_instance in self.polygons:
            polygon_points = np.concatenate(polygons_per_instance, axis=0).reshape(-1, 2)
            # 1. Convet polygon to convex hull.
            hull = cv2.convexHull(polygon_points.astype(np.float32), clockwise=False)
            # (N, 1, 2)
            if hull.shape[0] < 3:
                continue

            hull = hull.reshape(-1, 2)

            try:
                dilated_hull = dilate_polygon_by_rate(hull, dilate_rate)
                 #dilated_hull = dilate_polygon(hull, np.sqrt(polygon_area(hull[:, 0], hull[:, 1])) * expansion_ratio)
            except IndexError:
                continue
            instance_width = int(dilated_hull[:, 0].max() - dilated_hull[:, 0].min() + 1 - 1e-5)
            instance_height = int(dilated_hull[:, 1].max() - dilated_hull[:, 1].min() + 1 - 1e-5)
            if instance_height < 2 or instance_width < 2:
                print('hull:')
                print(hull)
                print('dilated_hull')
                print(dilated_hull)
                continue
            center_mask_for_instance = np.zeros((instance_height, instance_width), dtype=np.float32)

            # Perform rendering on cropped areas to save computation cost.
            shift = dilated_hull.min(0)
            polygon_points = polygon_points - shift
            hull = hull - shift
            dilated_hull = dilated_hull - shift
            point_o = hull.mean(axis=0)

            center_mask_for_instance = mask_in_triangle_with_padding(
                hull,
                instance_width,
                instance_height,
                point_o,
                center_mask_for_instance,
                reverse=True,
                standard=standard,
                sigma=sigma,
                dilate_rate=dilate_rate,
                neg_points_thresh=neg_points_thresh)

            instance_mask_offset.append(shift)
            instances_center_mask.append(center_mask_for_instance)
            # 4. Attach to the mask for whole image
            xmin, ymin = shift
            xmax = xmin + instance_width
            ymax = ymin + instance_height
            xmin_valid = min(max(0, xmin), center_mask.shape[1] - 1)
            xmax_valid = min(max(0, xmax), center_mask.shape[1] - 1)
            ymin_valid = min(max(0, ymin), center_mask.shape[0] - 1)
            ymax_valid = min(max(0, ymax), center_mask.shape[0] - 1)

            center_mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid] = np.fmax(
                center_mask_for_instance[
                ymin_valid - ymin:ymax_valid - ymax + instance_height,
                xmin_valid - xmin:xmax_valid - xmax + instance_width],
                center_mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid])

        return center_mask, instances_center_mask, instance_mask_offset

    def masks(
            self,
            mask: np.ndarray = None,
            mask_size: Union[int, tuple, None] = None
    ) -> np.ndarray:
        """
        Generate masks inside polygons with gradient value.
        """

        assert not (mask and mask_size), "Only one of mask and size should be specified."
        if mask is None:
            mask_size = make_dual(mask_size)
        border_mask = np.zeros(mask_size, dtype=np.float32)
        center_mask = np.zeros(mask_size, dtype=np.float32)
        size_mask = np.zeros((*mask_size, 2), dtype=np.float32)
        border_mask, center_mask, size_mask = self.border_masks(
            border_mask, center_mask, size_mask)
        return border_mask, center_mask, size_mask

    def border_masks(
            self,
            border_mask: np.ndarray,
            center_mask: np.ndarray,
            size_mask: np.ndarray,
            expansion_ratio: float = 0.1
    ) -> np.ndarray:
        assert border_mask.shape == center_mask.shape

        for polygons_per_instance in self.polygons:
            polygon_points = np.concatenate(polygons_per_instance, axis=0).reshape(-1, 2)
            # 1. Convet polygon to convex hull.
            hull = cv2.convexHull(polygon_points.astype(np.float32), clockwise=False)
            # (N, 1, 2)
            if hull.shape[0] < 3:
                continue

            hull = hull.reshape(-1, 2)

            try:
                dilated_hull = dilate_polygon(hull, np.sqrt(polygon_area(hull[:, 0], hull[:, 1])) * expansion_ratio)
            except IndexError:
                continue
            instance_width = int(dilated_hull[:, 0].max() - dilated_hull[:, 0].min() + 1 - 1e-5)
            instance_height = int(dilated_hull[:, 1].max() - dilated_hull[:, 1].min() + 1 - 1e-5)
            border_mask_for_instance = np.zeros((instance_height, instance_width), dtype=np.float32)
            center_mask_for_instance = np.zeros((instance_height, instance_width), dtype=np.float32)
            size_mask_for_instance = np.zeros((instance_height, instance_width, 2), dtype=np.float32)

            # Perform rendering on cropped areas to save computation cost.
            shift = dilated_hull.min(0)
            polygon_points = polygon_points - shift
            hull = hull - shift
            dilated_hull = dilated_hull - shift
            point_o = hull.mean(axis=0)

            # 2. Draw l2 distance_map 
            draw_border_map(hull, border_mask_for_instance, expansion_ratio)

            # 3. Draw l1 distance in areas for each neighboring point pairs
            border_mask_for_instance = mask_in_triangle(
                hull,
                instance_width,
                instance_height,
                point_o,
                border_mask_for_instance,
                reverse=False)
            center_mask_for_instance = mask_in_triangle(
                hull,
                instance_width,
                instance_height,
                point_o,
                center_mask_for_instance,
                reverse=True)
            size_mask_for_instance = distance_in_triangle(hull, point_o, size_mask_for_instance)
            # 4. Attach to the mask for whole image
            xmin, ymin = shift
            xmax = xmin + instance_width
            ymax = ymin + instance_height
            xmin_valid = min(max(0, xmin), border_mask.shape[1] - 1)
            xmax_valid = min(max(0, xmax), border_mask.shape[1] - 1)
            ymin_valid = min(max(0, ymin), border_mask.shape[0] - 1)
            ymax_valid = min(max(0, ymax), border_mask.shape[0] - 1)

            border_mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid] = np.fmax(
                border_mask_for_instance[
                ymin_valid - ymin:ymax_valid - ymax + instance_height,
                xmin_valid - xmin:xmax_valid - xmax + instance_width],
                border_mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid])

            center_mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid] = np.fmax(
                center_mask_for_instance[
                ymin_valid - ymin:ymax_valid - ymax + instance_height,
                xmin_valid - xmin:xmax_valid - xmax + instance_width],
                center_mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid])

            size_mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid, :] = np.fmax(
                size_mask_for_instance[
                ymin_valid - ymin:ymax_valid - ymax + instance_height,
                xmin_valid - xmin:xmax_valid - xmax + instance_width,
                :],
                size_mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid, :])

        return border_mask, center_mask, size_mask

    def gradient(self, mask):
        horizontal_padding = np.concatenate(
            [np.zeros((mask.shape[0], 1), dtype=mask.dtype), mask],
            axis=1)
        vertical_padding = np.concatenate(
            [np.zeros((1, mask.shape[1]), dtype=mask.dtype), mask],
            axis=0)

        return np.abs(horizontal_padding[:, 1:] - horizontal_padding[:, :-1]), \
               np.abs(vertical_padding[1:] - vertical_padding[:-1])
