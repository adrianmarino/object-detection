import cv2

from lib.stream.image_frame import ImageFrame


class ImageReader:
    def __init__(self, paths):
        self.__paths = paths
        self.__index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__index < len(self.__paths):
            path = self.__paths[self.__index]
            self.__index += 1

            image = cv2.imread(path)
            height, width = image.shape[:2]

            return ImageFrame(image, width, height, path)
        else:
            raise StopIteration

    def close(self):
        pass
