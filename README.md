#  object-detection


* Detect objects from image:

    ```bash
    bin/predictor \
        --model-path models/inference_graph_106421/frozen_inference_graph.pb \
        --label-map-path dataset/label_map.pbtxt \
        --input-image examples/never_seen_sample_1 \
        --output .
    ```

* Detect objects from webcam:

    ```bash
    bin/predictor \
        --model-path models/inference_graph_106421/frozen_inference_graph.pb \
        --label-map-path dataset/label_map.pbtxt \
        --input-webcam 0 \
        --output ./video.mp4
    ```

* Press **esc**/**q** key to end process.
