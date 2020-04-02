import cv2
import os

from lib.stream.writer.writter import Writer


class ImageWriter(Writer):
    def __init__(self, path):
        self.__path = path

    def write(self, frame):
        output_path = os.path.join(self.__path, frame.filename)
        cv2.imwrite(output_path, frame.raw)

    def close(self):
        pass
