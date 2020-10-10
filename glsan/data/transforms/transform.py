import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)
from PIL import Image

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

__all__ = [
    "ResizePaddingTransform",
]


class ResizePaddingTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, min_size, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            min_size (int): min_size for each edge
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4

        if img.dtype == np.uint8:
            pil_image = Image.fromarray(img)
            interp_method = interp if interp is not None else self.interp
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)

            # padding
            if self.new_h < self.min_size:
                pad = self.min_size - self.new_h
                ret = cv2.copyMakeBorder(ret, 0, pad, 0, 0, cv2.BORDER_CONSTANT)
            elif self.new_w < self.min_size:
                pad = self.min_size - self.new_w
                ret = cv2.copyMakeBorder(ret, 0, 0, 0, pad, cv2.BORDER_CONSTANT)
            return ret
        else:
            # PIL only supports uint8
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {Image.BILINEAR: "bilinear", Image.BICUBIC: "bicubic"}
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            img = F.interpolate(img, (self.new_h, self.new_w), mode=mode, align_corners=False)
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

            # padding
            if self.new_h < self.min_size:
                pad = self.min_size - self.new_h
                ret = cv2.copyMakeBorder(ret, 0, pad, 0, 0, cv2.BORDER_CONSTANT)
            elif self.new_w < self.min_size:
                pad = self.min_size - self.new_w
                ret = cv2.copyMakeBorder(ret, 0, 0, 0, pad, cv2.BORDER_CONSTANT)
            return ret

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)
