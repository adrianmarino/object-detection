import logging

import numpy as np
import tensorflow as tf

from lib.tf_od_api.models.research.object_detection.utils import label_map_util
from lib.tf_od_api.models.research.object_detection.utils import visualization_utils as vis_util


def load_inference_graph(path):
    logging.info(f'Loading {path} model...')
    graph = tf.compat.v1.Graph()

    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path, 'rb') as fid:
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


class Model:
    def __init__(self, inference_graph_path, label_map_path, classes):
        self.graph = load_inference_graph(inference_graph_path)
        self.__category_index = get_category_index(label_map_path, classes)

    def predict(self, session, frame):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame.raw, axis=0)

        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')

        num_detections = self.graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = session.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame.raw,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.__category_index,
            use_normalized_coordinates=True,
            line_thickness=1,
            max_boxes_to_draw=50,
            min_score_thresh=.25
        )
