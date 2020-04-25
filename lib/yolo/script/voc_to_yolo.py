import argparse
import xml.etree.ElementTree as ET
from glob import glob

from lib.config import Config
from lib.logger_factory import LoggerFactory

class VOCBoundingBox:

    @staticmethod
    def from_obj(obj):
        xmlbox = obj.find('bndbox')
        return (
            float(xmlbox.find('xmin').text),
            float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymin').text),
            float(xmlbox.find('ymax').text)
        )

    @staticmethod
    def to_yolo(bounding_box, image_size):
        dw = 1. / image_size[0]
        dh = 1. / image_size[1]

        x = (bounding_box[0] + bounding_box[1]) / 2.0
        y = (bounding_box[2] + bounding_box[3]) / 2.0
        w = bounding_box[1] - bounding_box[0]
        h = bounding_box[3] - bounding_box[2]

        x = x * dw
        y = y * dh

        w = w * dw
        h = h * dh
        return (x, y, w, h)


class YOLOBoundingBox:

    @staticmethod
    def to_record(cls_id, bounding_box):
        return f"{cls_id} {bounding_box[0]:0.6f} {bounding_box[1]:0.6f} {bounding_box[2]:0.6f} {bounding_box[3]:0.6f}\n"


def convert_annotation(voc_file_path):
    with open(voc_file_path) as in_file:
        txt_fn = voc_file_path.replace(".xml", ".txt")
        with open(txt_fn, 'w') as out_file:
            root = get_root(in_file)
            image_size = get_image_size(root)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text

                if cls not in classes or int(difficult) == 1:
                    continue

                cls_id = classes.index(cls)

                voc_bounding_box = VOCBoundingBox.from_obj(obj)
                yolo_bounding_box = VOCBoundingBox.to_yolo(voc_bounding_box, image_size)
                yolo_record = YOLOBoundingBox.to_record(cls_id, yolo_bounding_box)

                out_file.write(yolo_record)


def get_root(in_file):
    tree = ET.parse(in_file)
    root = tree.getroot()
    return root


def get_image_size(root):
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    image_size = (w, h)
    return image_size


class InputParamsResolver:
    def __init__(self):
        parser = argparse.ArgumentParser(prog="VOC to YOLO", description='A VOC to YOLO samples converter')
        parser.add_argument('--input-path', help='Input VOC samples path.')
        parser.add_argument('--resume-file', help='File where write all images paths.')
        self.__parser = parser

    @property
    def params(self):
        return {k: v for k, v in dict(self.__parser.parse_args()._get_kwargs()).items() if v is not None}


if __name__ == '__main__':
    config = Config('./config.yml')
    logger = LoggerFactory(config['logger']).create()
    resolver = InputParamsResolver()

    classes = config.property('labels')
    logger.info(f'Classes {len(classes)}: {classes} ')

    with open(resolver.params['resume_file'], "w") as resume_file:
        for i, voc_file_path in enumerate(glob(f'{resolver.params["input_path"]}/*.xml')):
            convert_annotation(voc_file_path)

            img_file_path = voc_file_path.replace(".xml", ".jpg")
            resume_file.write(f'{img_file_path}\n')

            logger.info(f'{voc_file_path} processed')
