import os.path
import pathlib
import sys
import tarfile

import cv2
import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf

cap = cv2.VideoCapture(2)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from lib.tensorflow_object_detection_api.research.object_detection.utils import label_map_util
from lib.tensorflow_object_detection_api.research.object_detection.utils import visualization_utils as vis_util

# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:
# In[4]:

# What model to download.

print('\n\nSSD Models:')
print('- ssd_mobilenet_v1_coco_11_06_2017')
print('- ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03')
print('- ssd_inception_v2_coco_2018_01_28')
print('- ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03')
print('\n\n')
print('Faster RCNN Models:')
print('- faster_rcnn_nas_coco_2018_01_28')
print('- faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28')
print('- faster_rcnn_resnet101_fgvc_2018_07_19')
print('- faster_rcnn_resnet50_coco_2018_01_28')
print('\n\n')
print('Mask RCNN Models:')
print('- mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28')
print('\n\n')


DEFAULT_MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

if len(sys.argv) > 1 and len(sys.argv[1]) > 0:
    MODEL_NAME = str(sys.argv[1])
    print(f'==> Selected model: {MODEL_NAME} <==\n\n')
else:
    print(f'==> Set default model: {DEFAULT_MODEL_NAME} <==\n\n')
    MODEL_NAME = DEFAULT_MODEL_NAME

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('lib', 'tensorflow_object_detection_api', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model

# In[5]:

if not pathlib.Path(f'./{MODEL_NAME}').exists():
    print(f'Downloading {MODEL_NAME} model...')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
else:
    print(f'{MODEL_NAME} model already downloaded...')

print(f'Loading {MODEL_NAME} model...')
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

print(f'Loading labels from {PATH_TO_LABELS}...')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# In[10]:

print('Begin model evaluation...')
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

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1,
                max_boxes_to_draw=30,
                min_score_thresh=.45
            )

            cv2.imshow('object detection', cv2.resize(image_np, (1280, 720)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
