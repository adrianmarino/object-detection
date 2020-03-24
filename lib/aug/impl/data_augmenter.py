import logging
import os
from multiprocessing.pool import ThreadPool

import cv2

from lib.aug.dataset_augmenter import IDataAugmenter
from lib.dataset.image import Image
from lib.dataset.pascal_voc.pascal_voc_file import PascalVOCFile
from lib.dataset.sample import Sample
from lib.list_utils import flatten
from lib.util import os_utils

logger = logging.getLogger()


class DataAugmenter(IDataAugmenter):
    def __init__(self, dataset, output_path, max_parallel_augs=50):
        self.__output_path = output_path
        os_utils.create_path(output_path)
        self.__dataset = dataset
        self.max_parallel_augs = max_parallel_augs

    def augment(self, samples, aug_strategy, aug_count=10):
        with ThreadPool(self.max_parallel_augs) as pool:
            return flatten(
                pool.starmap_async(
                    self.sample_augment,
                    [(aug_strategy, s, s_num, aug_count) for s_num, s in enumerate(samples, start=1) if
                     s.bounding_boxes is not None and len(s.bounding_boxes) > 0]
                ).get()
            )

    def sample_augment(self, aug_strategy, sample, sample_num, aug_count):
        max_pool = self.max_parallel_augs if aug_count > self.max_parallel_augs else aug_count

        logger.info(f'>>> {sample_num} sample <<< - File: {sample.image.filename()}...')

        with ThreadPool(max_pool) as pool:
            return pool.starmap_async(
                self.sample_augment_time,
                [(aug_strategy, sample, index) for index in range(0, aug_count)]
            ).get()

    def sample_augment_time(self, aug_strategy, sample, index):
        input_image_path = f'{self.__dataset.path}/{sample.image.filename()}'

        if not os.path.exists(input_image_path):
            raise Exception(f'Not found {input_image_path} image path!')

        aug_image_path = self.destiny_path(sample.image, index)
        aug_xml_path = os.path.splitext(aug_image_path)[0] + '.xml'
        if os.path.exists(aug_image_path) and os.path.exists(aug_xml_path):
            logger.info(f'Already exist {os.path.basename(aug_image_path)} augmented sample...')
            return PascalVOCFile.load(aug_xml_path)

        original_raw_image = cv2.imread(input_image_path)

        aug_raw_image, aug_bboxes = aug_strategy.augment(
            original_raw_image,
            sample.bounding_boxes
        )

        aug_sample = self.__create_sample(sample.image, aug_raw_image, aug_bboxes, index)

        cv2.imwrite(aug_sample.image.path, aug_raw_image)
        PascalVOCFile.write(aug_sample)

        return aug_sample

    def __create_sample(self, img, aug_raw_image, aug_bboxes, index):
        image = Image(
            self.destiny_path(img, index),
            height=aug_raw_image.shape[0],
            width=aug_raw_image.shape[1],
            depth=img.depth
        )
        return Sample(image, aug_bboxes)

    def destiny_path(self, img, index):
        parts = img.filename().split('.')
        return f'{self.__output_path}/{parts[0]}{index}.{parts[1]}'
