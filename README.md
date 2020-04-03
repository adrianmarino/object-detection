#  object-detection


* Detect objects from image:

    ```bash
    bin/predictor \
        --model-path models/inference_graph_16232/frozen_inference_graph.pb \
        --label-map-path dataset/label_map.pbtxt \
        --input-image ./input/never_seen_sample_4.jpg \
        --output ./output
    ```

* Detect objects from webcam:

    ```bash
    bin/predictor \
        --model-path models/inference_graph_16232/frozen_inference_graph.pb \
        --label-map-path dataset/label_map.pbtxt \
        --input-webcam 2 \
        --output ./output/video.mp4
    ```

* Detect objects from video file:

    ```bash
    bin/predictor \
        --model-path models/inference_graph_16232/frozen_inference_graph.pb \
        --label-map-path dataset/label_map.pbtxt \
        --input-video input/test_video_2.mp4 \
        --output ./output/test_video_2.mp4 \
        --show-preview \
        --preview-scale 50
    ```

* Press **q** key to end process.
