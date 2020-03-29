import glob

import cv2
import numpy as np


def available_video_ports():
    ports = [filename.split('video')[1] for filename in glob.glob('/dev/video*')]
    ports.sort()
    available_ports = []

    for port in ports:
        try:
            video_capture = cv2.VideoCapture(int(port))
            if video_capture.read()[0]:
                available_ports.append(port)
            video_capture.release()
        except:
            continue

    return available_ports


def assert_video_port_availability(video_port):
    available_ports = available_video_ports()
    is_available = str(video_port) in available_ports
    assert is_available, f"Video port {video_port} is not available!. Use any of these: {' or '.join(available_ports)}."
    return video_port


class VideoReader:
    def __init__(self, video_port=0):
        self.__reader = cv2.VideoCapture(video_port)

    def next(self, flip=False):
        has_frame, frame = self.__reader.read()
        return has_frame, cv2.flip(frame.copy(), 1) if flip else frame

    def size(self):
        return round(self.width()), round(self.height())

    def width(self):
        return self.__reader.get(cv2.CAP_PROP_FRAME_WIDTH)

    def height(self):
        return self.__reader.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def fps(self):
        return self.__reader.get(cv2.CAP_PROP_FPS)

    def close(self):
        self.__reader.release()


class VideoWriter:
    def __init__(self, path, fps, size):
        self.writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            fps,
            size
        )

    def write(self, frame): self.writer.write(frame.astype(np.uint8))

    def close(self): self.writer.release()
