import argparse
import logging

import cv2
import numpy as np
import tensorflow as tf

from lib.config import Config
from lib.tf_od_api.models.research.object_detection.utils import label_map_util
from lib.tf_od_api.models.research.object_detection.utils import visualization_utils as vis_util

logging.getLogger().setLevel(logging.INFO)


def get_params():
    parser = argparse.ArgumentParser(description='Predictor.')
    parser.add_argument(
        '--model-path',
        help='Path of final model graph'
    )
    parser.add_argument(
        '--label-map-path',
        help='Path of label map file'
    )
    return parser.parse_args()


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
    category_index = label_map_util.create_category_index(categories)
    logging.info(f'Category Index: {category_index}')
    return category_index


if __name__ == '__main__':
    params = get_params()
    cfg = Config('./config.yml')
    classes = cfg.property('labels')

    detection_graph = load_model_state(params.model_path)
    category_index = get_category_index(params.label_map_path, classes)

    cap = cv2.VideoCapture(2)

    logging.info('Begin model evaluation...')
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, image_np = cap.read()

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
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

                logging.info(f'Classes: {classes}')

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=1,
                    max_boxes_to_draw=60,
                    min_score_thresh=.10
                )

                scale_percent = 300  # percent of original size
                width = int(image_np.shape[1] * scale_percent / 100)
                height = int(image_np.shape[0] * scale_percent / 100)
                dim = (width, height)

                # resize image
                resized = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow('Object detection', resized)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
