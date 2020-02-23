import os

import untangle

from lib.list_utils import group, to_list, flatten, by_field_name
from lib.pascal_voc.bounding_box import BoundingBox
from lib.pascal_voc.image_size import Image
from lib.pascal_voc.sample import Sample


class Dataset:
    def __init__(self, path):
        self.path = path

    def samples(self):
        for filename in os.listdir(self.path):
            if not filename.endswith('.xml'):
                continue
            root = self.__load_xml(filename)
            image = self.__create_image(root)
            bounding_boxes = [self.__create_bounding_box(obj) for obj in to_list(root.object)]
            yield Sample(image, bounding_boxes)

    def __load_xml(self, filename):
        filename = os.path.join(self.path, filename)
        root = untangle.parse(filename)
        return root.annotation

    def __create_image(self, root):
        return Image(
            root.path.cdata,
            int(root.size.width.cdata),
            int(root.size.height.cdata),
            int(root.size.depth.cdata)
        )

    def __create_bounding_box(self, obj):
        return BoundingBox(
            obj.name.cdata,
            int(obj.bndbox.xmin.cdata),
            int(obj.bndbox.ymin.cdata),
            int(obj.bndbox.xmax.cdata),
            int(obj.bndbox.ymax.cdata)
        )

    def group_bounding_boxes_by(self, field_name):
        return group(self.bounding_boxes(), by_field_name(field_name))

    def bounding_boxes(self):
        return flatten(map(lambda it: it.bounding_boxes, self.samples()))

    def classes(self):
        return self.group_bounding_boxes_by('class_name').keys()