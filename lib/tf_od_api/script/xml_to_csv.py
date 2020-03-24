import argparse
import glob
import logging
import os

import pandas as pd

from lib.dataset.pascal_voc.pascal_voc_file import PascalVOCFile

logging.getLogger().setLevel(logging.INFO)


def to_samples_dt(path, setname, dataset_path):
    rows = []

    data_file_paths = glob.glob(path + '/*.xml')
    logging.info(f'Samples count: {len(data_file_paths):.0f}')

    fails_count = 1
    for data_file_path in data_file_paths:
        if fails_count > 10:
            logging.error(f'More than {fails_count} inconsistent samples!. Process aborted!')
            exit(1)

        imagepath = get_image_path(data_file_path)
        if not os.path.isfile(imagepath):
            logging.warning(
                f'Ignore {os.path.basename(data_file_path)} sample because does not exist {os.path.basename(imagepath)} image file!')
            fails_count += 1
            continue

        try:
            sample = PascalVOCFile.load(data_file_path)

            for bb in sample.bounding_boxes:
                rows.append(to_row(bb, sample, setname, dataset_path))
        except Exception as error:
            logging.error(error)
            logging.warning(f'Error to process {data_file_path} from {setname} set. Sample excluded!.')
            fails_count += 1

    return create_table(rows)


def get_image_path(xml_file):
    return os.path.splitext(xml_file)[0] + '.jpg'


def create_table(rows):
    return pd.DataFrame(
        rows,
        columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    )


def to_row(bb, sample, setname, dataset_path):
    return (
        os.path.join(dataset_path, setname, 'samples', os.path.basename(sample.image.path)),
        sample.image.width,
        sample.image.height,
        bb.label,
        bb.xmin,
        bb.ymin,
        bb.xmax,
        bb.ymax
    )


def get_params():
    parser = argparse.ArgumentParser(description='XML to CSV.')
    parser.add_argument(
        '--dataset-path',
        help='Dataset path.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    params = get_params()

    logging.info(f'Dataset path: {params.dataset_path}')

    for set_name in ['train', 'test']:
        image_path = os.path.join(params.dataset_path, set_name, 'samples')

        output_file_path = os.path.join(params.dataset_path, set_name, f'{set_name}_labels.csv')

        logging.info(f'Generate {set_name} samples csv in {output_file_path} path.')
        logging.info(f'Analising {image_path} path.')

        samples_dt = to_samples_dt(image_path, set_name, params.dataset_path)
        logging.info(f'Bounding boxes: {len(samples_dt):.0f}')

        samples_dt.to_csv(output_file_path, index=None)
        logging.info(f'{output_file_path} successfully generated!')
