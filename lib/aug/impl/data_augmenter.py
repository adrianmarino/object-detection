import os
from multiprocessing.pool import ThreadPool

import cv2

from lib.aug.dataset_augmenter import IDataAugmenter
from lib.dataset.image import Image
from lib.dataset.pascal_voc.pascal_voc_file import PascalVOCFile
from lib.dataset.sample import Sample
from lib.list_utils import flatten


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


class DataAugmenter(IDataAugmenter):
    def __init__(self, dataset, output_path, max_parallel_augs=8):
        self.__output_path = output_path
        create_directory(output_path)
        self.__dataset = dataset
        self.max_parallel_augs = max_parallel_augs

    def augment(self, samples, aug_strategy, aug_count=10):
        with ThreadPool(self.max_parallel_augs) as pool:
            return flatten(
                pool.starmap_async(self.sample_augment, [(aug_strategy, s, aug_count) for s in samples]).get()
            )

    def sample_augment(self, aug_strategy, sample, aug_count):
        max_pool = self.max_parallel_augs if aug_count > self.max_parallel_augs else aug_count

        with ThreadPool(max_pool) as pool:
            return pool.starmap_async(
                self.sample_augment_time,
                [(aug_strategy, sample, index) for index in range(0, aug_count)]
            ).get()

    def sample_augment_time(self, aug_strategy, sample, index):
        input_image_path = f'{self.__dataset.path}/{sample.image.filename()}'

        if not os.path.exists(input_image_path):
            raise Exception(f'Not found {input_image_path} image path!')

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
        parts = img.filename().split('.')
        path = f'{self.__output_path}/{parts[0]}{index}.{parts[1]}'
        image = Image(
            path,
            height=aug_raw_image.shape[0],
            width=aug_raw_image.shape[1],
            depth=img.depth
        )
        return Sample(image, aug_bboxes)
