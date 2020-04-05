#  object-detection

Detect objects using [Tensorflow Object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Setup

**Step 1**: Create project environment.

```bash
conda env create --file environment.yml
```

**Step 2**: Donwload [Tensorflow Object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

```bash
git clone https://github.com/tensorflow/models.git lib/tf_od_api/models
```

This download models repository to `lib/tf_od_api` path.

## Training

**Step 1**: First of all must generate a dataset with train(70%) & test(30%) samples. 
There are many ways to generate a dataset:

1. From scratch taking photos from cellphone and then mark each 
bounding box under each image using [labelimg](https://github.com/tzutalin/labelImg). 
Maybe this is the more straightforward way to generate a dataset, but not the best way to
reach state of the art results. The problem is that it take a lot of time to make a tiny 
and consistent dataset.

2. Other way to generate a dataset could be take a photo of each object class and then
use this to generate new examples that combine this classes or not. Also can apply many transformations 
and filters but this totally depends on the domain of the problem to solve. This process is called data
 augmentation and can use a tool like [imgaug](https://github.com/aleju/imgaug). These option
 has better results but you have to choose the transformations and filters with a lot of criteria.

The dataset must have next structure:

```bash
./dataset/train/samples
    sample1.jpg
    sample1.xml
     ....
    sampleN.jpg
    sampleN.xml <---- Pascal VOC file
./dataset/train/samples
    sample1.jpg
    sample1.xml
    ....
    sampleN.jpg
    sampleN.xml <---- Pascal VOC file
```

Pascal VOC files contains position, size and class for each bounding box you want to infer
from an image. This files are generated by [labelimg](https://github.com/tzutalin/labelImg) 
tool but also you can create these from scratch.


**Step 2**: Create `label_map.pbtxt` file under `dataset` path. This file map class names 
from Pascal VOC files to integer values. Add next items for each class to `label_map.pbtxt` file:


```bash
item {
	id: 1
	name: 'Class1'
}
item {
	id: 2
	name: 'Class2'
}
...
item {
	id: N
	name: 'ClassN'
}
```

**Step 3**: Select a model to train from [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 
For example, download [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)
and extract this under as a directory with `training` name.


**Step 4**: Config next properties in `pipeline.config`:

```bash
train_config {
  ...
  fine_tune_checkpoint: "/PATH/TO/training/model.ckpt"
  ...
}

train_input_reader {
  label_map_path: "/PATH/TO/dataset/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "/PATH/TO/training/train.record"
  }
}

eval_config {
  ...
  num_examples: 3000 <= Numer of samples under dataset/test/samples.
  ...
}

eval_input_reader {
  label_map_path: "/PATH/TO/dataset/label_map.pbtxt"
  ...  
  tf_record_input_reader {
    input_path: "/PATH/TO/training/test.record"
  }
} 
```

**Step 5**: Remove `training/checkpoint` file.


**Step 6**: Generate train.record and test.record files under training path. This files contains
all samples data and images. Run next command to perform this task:

```bash
bin/prepare-dataset $(pwd)/dataset
```

**Step 7**: Read [Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md) to understand who to make model tuning.

 
**Step 8**: Train model.
 
```bash
bin/train 500000
```
 
_Note_: 500.000 is the number of steps. An steps represent the number of batches of samples to process.

To see training output can use:

```bash
bin/train-output
```

**Step 9**: To check the accuracy of model must use mAP, AR and F1 Score metrics. 
You can check this from a tensorboard. To run tensorboard:


```bash
bin/train-monitor
```

**Step 10**: Go to [http://localhost:6006](http://localhost:6006/) url.


**Step 11**: After train model you must export inference graph to `model` directory. Select a checkpoint number from training path:

```bash
$ ls -l training/model.ckpt-*.meta
-rw-r--r-- 1 adrian adrian 15929850 Apr  3 14:42 training/model.ckpt-22718.meta
-rw-r--r-- 1 adrian adrian 15929850 Apr  3 14:52 training/model.ckpt-23162.meta
-rw-r--r-- 1 adrian adrian 15929850 Apr  3 15:02 training/model.ckpt-23606.meta
-rw-r--r-- 1 adrian adrian 15929850 Apr  3 15:12 training/model.ckpt-24050.meta
-rw-r--r-- 1 adrian adrian 15929850 Apr  3 15:22 training/model.ckpt-24494.meta
```

And next run export-inference-graph script:


```bash
bin/export-inference-graph 24494
```


## Notebooks

* [dataset-analisys.ipynb](notebooks/dataset-analisys.ipynb): Check that dataset contains all classes. Also check that dataset is balance.
* [creating-playing-cards-dataset.ipynb](notebooks/dataset-generation/creating-playing-cards-dataset.ipynb): Create a poker cards dataset using augmentation.
* [data-aumentation.ipynb](/notebooks/data-augmenter.ipynb): Augment dataset samples using 'DataAugmenter' class, that is an abstraction under [imgaug](https://github.com/aleju/imgaug).
* [f1-score-metric.ipynb](/notebooks/metrics/f1-score-metric.ipynb): Calculate **F1 SCore** metric useful to check object detection model accuracy.

## Prediction

**Step 1:**: Activate environment.

```bash
source object-detection
```

**Step 2:**:

    * Detect objects on images:

        ```bash
        bin/predictor \
            --model-path models/inference_graph_16232/frozen_inference_graph.pb \
            --label-map-path dataset/label_map.pbtxt \
            --input-image ./input/never_seen_sample_4.jpg \
            --output ./output
        ```
    
    * Detect objects on video files:
    
        ```bash
        bin/predictor \
            --model-path models/inference_graph_16232/frozen_inference_graph.pb \
            --label-map-path dataset/label_map.pbtxt \
            --input-video input/test_video_2.mp4 \
            --output ./output/test_video_2.mp4 \
            --show-preview \
            --preview-scale 50
        ```
    
    * Detect objects from webcam:
    
        ```bash
        bin/predictor \
            --model-path models/inference_graph_16232/frozen_inference_graph.pb \
            --label-map-path dataset/label_map.pbtxt \
            --input-webcam 2 \
            --output ./output/video.mp4
        ```


**Step 3:**:  Press **q** key to end process.

**Step 4**: Use `--help` param to check all param options:


```bash
$ bin/predictor --help        

usage: object-detection-predictor [-h] [--model-path MODEL_PATH]
                                  [--label-map-path LABEL_MAP_PATH]
                                  [--input-image INPUT_IMAGE]
                                  [--input-video INPUT_VIDEO]
                                  [--input-webcam INPUT_WEBCAM]
                                  [--show-preview] [--disable-bboxes]
                                  [--preview-scale PREVIEW_SCALE]
                                  [--output OUTPUT]

Object detection predictor

optional arguments:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path of final model graph
  --label-map-path LABEL_MAP_PATH
                        Path of label map file
  --input-image INPUT_IMAGE
                        Input image path.
  --input-video INPUT_VIDEO
                        Input video path.
  --input-webcam INPUT_WEBCAM
                        Input video port. Available ports: [0, 2] (Detected &
                        non-used /dev/videoX port).
  --show-preview        Force show preview window.
  --disable-bboxes      Force disable bounding boxes
  --preview-scale PREVIEW_SCALE
                        Change preview scale. Default: 100
  --output OUTPUT       Output image/video path.
```