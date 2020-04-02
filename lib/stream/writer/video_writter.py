import cv2
import numpy as np

from lib.stream.writer.writter import Writer


class VideoWriter(Writer):
    def __init__(self, path, fps=30):
        self.__path = path
        self.__fps = fps
        self.__writer = None

    def __get_writer(self, size):
        if self.__writer is None:
            self.__writer = cv2.VideoWriter(
                self.__path,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                self.__fps,
                size
            )
        return self.__writer

    def write(self, frame):
        writer = self.__get_writer(frame.size)
        writer.write(frame.raw.astype(np.uint8))

    def close(self):
        self.__writer.release()
