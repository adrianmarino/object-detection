import time
import os
import sys
import logging

sys.path.append("..")

from lib.data_aug.data_augmenter import DataAugmenter
from lib.dataset.pascal_voc.dataset import Dataset
from lib.data_aug.data_aug import RandomHSV, RandomRotate, RandomScale, RandomShear, RandomTranslate, Resize, Sequence

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

SOURCE_DATASET = './data_aug/dataset1'
OUTPUT_PATH = f'{os.getcwd()}/data_aug/output'

def test_aug_dataset():
    # Prepare
    dataset = Dataset(SOURCE_DATASET)
    samples = list(dataset.samples())
    data_augmenter = DataAugmenter(
        dataset,
        OUTPUT_PATH, 
        [
            Resize(600),
            RandomHSV(40, 40, 30), 
            RandomScale(0.3, diff = True), 
            RandomTranslate(0.1, diff = True),
            RandomShear(0.4),
            RandomRotate(30)
        ]
    )

    # Perform
    augmented_samples = list(data_augmenter.augment(samples, augment_count=100))

    # Asserts
    assert len(augmented_samples) == 100