import imgaug.augmentables as iaa


class BBoxesOnImgFactory:

    @staticmethod
    def from_sample(sample):
        return iaa.BoundingBoxesOnImage(
            [BBoxesOnImgFactory.__to_img_aug__bbox(bb) for bb in sample.bounding_boxes],
            shape=(sample.image.height, sample.image.width)
        )

    @staticmethod
    def create(img_shape, bboxes):
        return iaa.BoundingBoxesOnImage(
            [BBoxesOnImgFactory.__to_img_aug__bbox(bb) for bb in bboxes],
            shape=img_shape
        )

    @staticmethod
    def __to_img_aug__bbox(bb):
        return iaa.BoundingBox(x1=bb.xmin, y1=bb.ymin, x2=bb.xmax, y2=bb.ymax, label=bb.label)
