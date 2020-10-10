import os

import torch
import torch.distributed as dist
from detectron2.layers import cat
import torch.distributed as dist

INF = 100000000


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def permute_to_N_HW_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, C, H, W) to (N, (HxW), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.permute(0, 2, 3, 1).reshape(N, -1, K)  # Size=(N, HxW, K)
    return tensor


def permute_predicts(cls_outs, embed_outs, embed_channels):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    cls_outs = [permute_to_N_HW_K(x, 1) for x in cls_outs]
    embed_outs = [permute_to_N_HW_K(x, embed_channels) for x in embed_outs]
    return cls_outs, embed_outs


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def compute_locations(shapes, strides, device):
    locations = []
    for level, (shape, stride) in enumerate(zip(shapes, strides)):
        h, w = shape
        locations_per_level = compute_locations_per_level(h, w, stride, device)
        locations.append(locations_per_level)

    return locations
