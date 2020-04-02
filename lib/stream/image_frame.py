import ntpath

import cv2


class ImageFrame:
    def __init__(self, raw, width, height, path=None):
        self.raw = raw
        self.width = width
        self.height = height
        self.size = (round(self.width), round(self.height))
        self.path = path
        self.filename = None if self.path is None else ntpath.basename(self.path)

    def __str__(self):
        return f'Frame size: {self.size}, path: {self.path}'

    def scale(self, scale_percent=100):
        width = int(self.width * scale_percent / 100)
        height = int(self.height * scale_percent / 100)

        # resize image
        resized_raw = cv2.resize(self.raw, (width, height), interpolation=cv2.INTER_AREA)
        return ImageFrame(resized_raw, width, height, self.path)
