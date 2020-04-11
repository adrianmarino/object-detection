import imgaug.augmentables as iaa


class BoundingBoxesOnImageFactory:

    @staticmethod
    def from_sample(sample):
        return BoundingBoxesOnImageFactory.create(
            bounding_boxes=sample.bounding_boxes,
            img_shape=(sample.image.height, sample.image.width)
        )

    @staticmethod
    def create(img_shape, bounding_boxes):
        return iaa.BoundingBoxesOnImage(
            bounding_boxes=[BoundingBoxesOnImageFactory.__to_img_aug__bbox(it) for it in bounding_boxes],
            shape=img_shape
        )

    @staticmethod
    def __to_img_aug__bbox(bb):
        return iaa.BoundingBox(x1=bb.xmin, y1=bb.ymin, x2=bb.xmax, y2=bb.ymax, label=bb.label)
