import time
import os
import sys
sys.path.append("..")

from lib.data_aug.data_augmenter import DataAugmenter
from lib.dataset.pascal_voc.dataset import Dataset
from lib.data_aug.data_aug import RandomHSV, RandomRotate, RandomScale, RandomShear, RandomTranslate, Sequence
import shutil


SOURCE_DATASET = './data_aug/dataset1'
OUTPUT_PATH = f'{os.getcwd()}/data_aug/output'

def test_aug_dataset():
    # Prepare
    dataset = Dataset(SOURCE_DATASET)
    samples = list(dataset.samples())
    data_augmenter = DataAugmenter(
        dataset,
        OUTPUT_PATH, 
        [RandomHSV(40, 40, 30), RandomScale(), RandomTranslate(), RandomShear()]
    )

    # Perform
    augmented_samples = list(data_augmenter.augment(samples, augment_count=5))

    # Asserts
    assert len(augmented_samples) == 5