import torch

from detectron2.structures import Boxes


def pairwise_dist(points: torch.Tensor, boxes: Boxes):
    """
    Given a points grid and a list of boxes, computer the distance between points and boxes.
    The results are normalized by box sizes.
    Args:
        points Tensor(N, 2): Array of N (x, y) points.
        boxes (Boxes):  Contains M boxes.
    Returns:
        Tensor: Distance, size (N, M).
    """

    # Mx2, 2
    box_centers = boxes.get_centers()
    boxes = boxes.tensor
    box_wh = boxes[:, 2:] - boxes[:, :2]

    # N, M, 2
    distance = (points[:, None] - box_centers[None]) / box_wh[None]

    del box_wh
    return distance.norm(dim=2)


def stride_match(strides: torch.Tensor, boxes: Boxes, maximum=64):
    """
    Determine if the boxes are at the same stride level with strides.
    Typically used in assigning gt boxes to FPN feature levels.
    Args:
        strides (Tensor): shape (N, ), indicating the strides.
        boxes (Boxes): M boxes.
    """
    boxes = boxes.tensor
    box_wh = boxes[:, 2:] - boxes[:, :2]
    box_strides = torch.pow(
        2,
        ((torch.log2(box_wh[:, 0]) + \
          torch.log2(box_wh[:, 1])) / 2).int(),
    ).clamp(strides.min(), strides.max())
    matched_matrix = torch.eq(strides[:, None], box_strides[None])
    return matched_matrix
