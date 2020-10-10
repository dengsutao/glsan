from detectron2.structures.masks import PolygonMasks as D2Polygon

from concern.support import ratio_of_bbox, ratio_of_polygon


class PolygonMasks(D2Polygon):
    def get_ratios(self, oriented=False):
        """
        Args:
            oriented bool: Use oriented boxes instead of bounding boxes.
        Returns:
            np.ndarray: h/w ratio of the min area rect around polygon masks.
        """
        if oriented:
            return [ratio_of_polygon(polygon) for polygon in self.polygons]
        return [ratio_of_bbox(bbox) for bbox in self.get_bounding_boxes()]
