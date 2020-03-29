#  object-detection


* Detect objects from image:

    ```bash
    bin/predictor \
        --model-path models/inference_graph_70628/frozen_inference_graph.pb \
        --label-map-path dataset/label_map.pbtxt \
        --input-image examples/IMG_20200225_121805.jpg \
        --output prediction.jpg
    ```

* Detect objects from webcam:

    ```bash
    bin/predictor \
        --model-path models/inference_graph_70628/frozen_inference_graph.pb \
        --label-map-path dataset/label_map.pbtxt \
        --input-webcam 0 \
        --output ./video.mp4
    ```

* Press **q** key to end process.
