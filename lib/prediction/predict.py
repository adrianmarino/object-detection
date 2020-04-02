import logging
import os

import cv2
import tensorflow as tf

from lib.config import Config
from lib.prediction.input_params import InputParamsResolver
from lib.prediction.model import Model
from lib.stream.reader.image_reader import ImageReader
from lib.stream.reader.video_reader import assert_video_port_availability, VideoReader
from lib.stream.writer.image_writter import ImageWriter
from lib.stream.writer.video_writter import VideoWriter

logging.getLogger().setLevel(logging.INFO)


def hide_tensorflow_logs(): os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_reader(params):
    if 'input_image' in params:
        return ImageReader(params['input_image'].split(','))
    elif 'input_webcam' in params:
        video_port = assert_video_port_availability(params['input_webcam'])
        return VideoReader(video_port)


def create_writer(params):
    if 'input_image' in params:
        return ImageWriter(params['output'])
    elif 'input_webcam' in params:
        return VideoWriter(params['output'])


def show(frame, scale_percent=200):
    if type(writer) is VideoWriter:
        cv2.imshow('Object detection', frame.scale(scale_percent))


def create_model(params):
    cfg = Config('./config.yml')
    return Model(
        params['model_path'],
        params['label_map_path'],
        classes=cfg.property('labels')
    )


def check_end_prediction_action_keys(): return cv2.waitKey(25) & 0xFF == ord('q')


if __name__ == '__main__':
    hide_tensorflow_logs()
    params = InputParamsResolver().resolve()
    model = create_model(params)
    reader, writer = create_reader(params), create_writer(params)

    with model.graph.as_default():
        with tf.compat.v1.Session(graph=model.graph) as session:
            for frame in reader:
                model.predict(session, frame)
                writer.write(frame)
                show(frame)
                if check_end_prediction_action_keys():
                    break

    cv2.destroyAllWindows()
    reader.close()
    writer.close()
