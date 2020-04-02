import glob

import cv2

from lib.stream.image_frame import ImageFrame


def available_video_ports():
    ports = [filename.split('video')[1] for filename in glob.glob('/dev/video*')]
    ports.sort()
    available_ports = []

    for port in ports:
        try:
            video_capture = cv2.VideoCapture(int(port))
            if video_capture.read()[0]:
                available_ports.append(port)
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
        self.__video_port = video_port
        self.__reader = None

    def __iter__(self):
        return self

    def __get_reader(self):
        if self.__reader is None:
            self.__reader = cv2.VideoCapture(self.__video_port)
        return self.__reader

    def __next__(self):
        has_frame, frame = self.__get_reader().read()
        if has_frame:
            return ImageFrame(
                frame,
                width=self.__reader.get(cv2.CAP_PROP_FRAME_WIDTH),
                height=self.__reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
        else:
            raise StopIteration

    def close(self):
        self.__get_reader().release()
