import logging
import sys

from lib.aug.data_aug_seq_factory import DataAugSequenceFactory
from lib.aug.impl.data_augmenter import DataAugmenter
from lib.aug.impl.img_aug_sample_aug_strategy import IMGAUGSampleAugStrategy
from lib.dataset.pascal_voc.pascal_voc_dataset import PascalVOCDataset

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

SOURCE_DATASET = 'test/aug/dataset1'
OUTPUT_PATH = 'test/aug/output'


def test_aug_dataset():
    # Prepare
    dataset = PascalVOCDataset(SOURCE_DATASET)
    samples = list(dataset.samples())
    data_augmenter = DataAugmenter(dataset, OUTPUT_PATH)

    strategy = IMGAUGSampleAugStrategy(DataAugSequenceFactory.sequence_a())

    # Perform
    aug_samples = data_augmenter.augment(samples, strategy, aug_count=3)

    # Asserts
    assert len(aug_samples) == 3
