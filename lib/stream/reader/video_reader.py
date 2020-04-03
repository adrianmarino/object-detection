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
        self.reader = cv2.VideoCapture(video_port)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.reader.read()

        if not ret:
            raise StopIteration
        if not self.reader.isOpened():
            raise StopIteration

        width = self.reader.get(3)
        height = self.reader.get(4)
        return ImageFrame(frame, width, height)

    def close(self):
        self.reader.release()
