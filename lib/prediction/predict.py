import logging
import os

import cv2
import numpy as np
import tensorflow as tf

from lib.config import Config
from lib.prediction.input_params import InputParamsResolver
from lib.prediction.video import VideoReader, assert_video_port_availability
from lib.prediction.video import VideoWriter
from lib.tf_od_api.models.research.object_detection.utils import label_map_util
from lib.tf_od_api.models.research.object_detection.utils import visualization_utils as vis_util

logging.getLogger().setLevel(logging.INFO)


def load_model_state(path):
    logging.info(f'Loading {path} model...')
    graph = tf.Graph()

    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return graph


def get_category_index(label_map_path, classes):
    logging.info(f'Loading labels from {label_map_path}...')
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=len(classes),
        use_display_name=True
    )
    return label_map_util.create_category_index(categories)


def hide_tensorflow_logs(): os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_reader(params):
    if 'input_image' in params:
        return VideoReader(params['input_image'])
    elif 'input_video' in params:
        return VideoReader(params['input_video'])
    elif 'input_webcam' in params:
        return VideoReader(assert_video_port_availability(params['input_webcam']))


def create_input_output():
    reader = create_reader(params)
    return reader, VideoWriter(params['output'], fps=reader.fps(), size=reader.size())


def write_output(video_writer, frame, params):
    if 'input_image' in params:
        cv2.imwrite(params['output'], frame)
    else:
        video_writer.write(frame)


def show_frame(frame, scale_percent=100):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Object detection', resized)


def prediction_bboxes(frame, detection_graph, category_index):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(frame, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1,
        max_boxes_to_draw=30,
        min_score_thresh=.5
    )


if __name__ == '__main__':
    hide_tensorflow_logs()
    params = InputParamsResolver().resolve()

    cfg = Config('./config.yml')
    classes = cfg.property('labels')

    detection_graph = load_model_state(params['model_path'])
    category_index = get_category_index(params['label_map_path'], classes)

    reader, writer = create_input_output()

    is_video = 'input-webcam' in params

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                has_frame, input_frame = reader.next(flip=is_video)
                if not has_frame:
                    break

                prediction_bboxes(input_frame, detection_graph, category_index)

                if is_video:
                    show_frame(input_frame, scale_percent=300)

                write_output(writer, input_frame, params)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

    cv2.destroyAllWindows()
    reader.close()
    writer.close()
