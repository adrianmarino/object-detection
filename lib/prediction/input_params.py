import argparse
import os

from lib.stream.reader.video_reader import available_video_ports


def intersection(lst1, lst2): return list(set(lst1) & set(lst2))


class NotFountFileError(Exception):
    def __init__(self, name, path): super(NotFountFileError, self).__init__(f'Error: Not found {path} {name} file!')


class InputParamsResolver:

    def __init__(self):
        parser = argparse.ArgumentParser(prog="object-detection-predictor", description='Object detection predictor')
        parser.add_argument('--model-path', help='Path of final model graph')
        parser.add_argument('--label-map-path', help='Path of label map file')
        parser.add_argument('--input-image', help='Path of an image file.')
        parser.add_argument('--output', help='Path of an output file.')
        parser.add_argument('--input-webcam', help=self.__input_webcam_info(), type=int, default=0)
        self.__parser = parser

    @staticmethod
    def __input_webcam_info():
        available_ports = available_video_ports()
        ports_info = f'Available ports: [{", ".join(available_ports)}]' if len(
            available_ports) > 0 else 'Not found active ports'
        return f"Input video port. {ports_info} (Detected & non-used /dev/videoX port)."

    @property
    def params(self):
        return {k: v for k, v in dict(self.__parser.parse_args()._get_kwargs()).items() if v is not None}

    @property
    def path_params(self):
        return {name: value for name, value in self.params.items() if name in ['input_image', 'input_video']}

    def resolve(self):
        for name, path in self.path_params.items():
            if not os.path.isfile(path):
                raise NotFountFileError(name, path)
        return self.params
