import imgaug.augmentables as iaa
import logging

from lib.aug.bounding_boxes_factory import BoundingBoxesOnImageFactory
from lib.aug.sample_aug_strategy import SampleAugStrategy
from lib.dataset.bounding_box import BoundingBox


class IMGAUGSampleAugStrategy(SampleAugStrategy):

    def __init__(self, sequence):
        self.__sequence = sequence
        self.logger = logging.getLogger()

    def augment(self, img, bboxes):
        bboxes = BoundingBoxesOnImageFactory.create(img.shape, bboxes)

        aug_img, aug_bboxes = self.__sequence(image=img, bounding_boxes=bboxes)

        return aug_img, [self.__to_domain_bbox(bb) for bb in aug_bboxes.bounding_boxes]

    @staticmethod
    def __to_domain_bbox(bb):
        return BoundingBox(bb.label, bb.x1, bb.y1, bb.x2, bb.y2)

    @staticmethod
    def __to_img_aug__bbox(bb):
        return iaa.BoundingBox(x1=bb.xmin, y1=bb.ymin, x2=bb.xmax, y2=bb.ymax, label=bb.label)
