import cv2
import os
import itertools

from lib.dataset.pascal_voc.bounding_box import BoundingBox
from lib.dataset.pascal_voc.sample import Sample
from lib.dataset.pascal_voc.image import Image

from lib.data_aug.data_aug import Sequence
import numpy as np
import shutil
from lib.dataset.pascal_voc.pascal_voc_file import PascalVOCFile


def value_to_idx_and_idx_to_value(elements, index_value=lambda it: it):
    index = 1
    class_to_idx = {}
    idx_to_class = {}
    for element in elements:
        value = index_value(element)
        class_to_idx[value] = index
        idx_to_class[index] = value  
        index +=1
    return class_to_idx, idx_to_class

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

class DataAugmenter:
    def __init__(self, dataset, output_path, sequence):
        self.__output_path = output_path
        create_directory(output_path)
        self.__dataset = dataset
        self.__sequence = Sequence(sequence)

    def augment(self, samples, augment_count):
        class_to_idx, idx_to_class = value_to_idx_and_idx_to_value(self.__dataset.classes())
        results = []

        for sample in samples:
            original_image = sample.image
            original_bboxes = self.__to_array(sample.bounding_boxes, class_to_idx)

            for index in range(0, augment_count):
                input_image_path = f'{self.__dataset.path}/{original_image.filename()}'
                original_raw_image = cv2.imread(input_image_path)

                #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
                original_raw_image = original_raw_image[:,:,::-1]

                augmented_image, augmented_bboxes = self.__sequence(original_raw_image, original_bboxes)
 
                augmented_sample = self.__create_sample(original_image, augmented_image, augmented_bboxes, idx_to_class, index) 

                cv2.imwrite(augmented_sample.image.path, augmented_image)
                PascalVOCFile.write(augmented_sample, augmented_sample.image.path)
 
                results.append(augmented_sample)
        return results

    def __create_sample(self, original_image, augmented_image, augmented_bboxes, idx_to_class, index):            
            parts = original_image.filename().split('.')
            print(parts)
            path = f'{self.__output_path}/{parts[0]}{index}.{parts[1]}'
            image = Image(
                path,
                augmented_image.shape[1],
                augmented_image.shape[0],
                original_image.depth
            )
            return Sample(image, self.__from_array(augmented_bboxes, idx_to_class))

    def __to_array(self, bounding_boxes, class_to_idx):
        return np.array([[it.xmin, it.ymin, it.xmax, it.ymax, class_to_idx[it.class_name]] for it in bounding_boxes], dtype=np.float16)

    def __from_array(self, bounding_boxes, idx_to_class):
        return [BoundingBox(idx_to_class[it[4]], it[0], it[1], it[2], it[3]) for it in bounding_boxes]
