import argparse

from lib.aug.data_aug_seq_factory import DataAugSequenceFactory
from lib.aug.impl.data_augmenter import DataAugmenter
from lib.aug.impl.img_aug_sample_aug_strategy import IMGAUGSampleAugStrategy
from lib.config import Config
from lib.dataset.pascal_voc.pascal_voc_dataset import PascalVOCDataset
from lib.logger_factory import LoggerFactory

cfg = Config('config.yml')
logger = LoggerFactory(cfg['logger']).create()


def get_params():
    parser = argparse.ArgumentParser(description='Dataset Augmenter.')
    parser.add_argument(
        '--input-path',
        help='Input samples path.'
    )
    parser.add_argument(
        '--output-path',
        help='Output samples path. IMPORTANT: Use a complete path!'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of augmentation workers One por input image. Default: 10.'
    )
    parser.add_argument(
        '--times',
        type=int,
        default=10,
        help='Number of times that must be augmented a single sample. Default: 10 times.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    params = get_params()

    logger.info(f'Input Path: {params.input_path}...')
    logger.info(f'Output Path: {params.output_path}...')
    logger.info(f'Augmentations per sample: {int(params.times)}...')
    logger.info(f'Augmentation workers: {int(params.workers)}...')

    dataset = PascalVOCDataset(params.input_path)

    data_augmenter = DataAugmenter(
        dataset,
        params.output_path,
        params.workers
    )
    sequence = DataAugSequenceFactory.sequence_b()
    sample_aug_strategy = IMGAUGSampleAugStrategy(sequence)

    input_samples = list(dataset.samples())
    logger.info(f'Input Samples: {len(input_samples)}...')
    logger.info(f'Output Samples: {len(input_samples)* int(params.times)}...')

    logger.info(f'Begin samples augmentation...')
    aug_samples = data_augmenter.augment(
        input_samples,
        sample_aug_strategy,
        aug_count=int(params.times)
    )

    logger.info(f'Finish samples augmentation...')
    logger.info(f'Augmented samples: {len(aug_samples)}')
