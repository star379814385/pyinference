from typing import Any, List

import cv2
import numpy as np

from .trt_base_inference import TensorRTInfer


class TimmClsTensorRTInfer(TensorRTInfer):
    def __init__(self, engine_path, num_buffer=1):
        super().__init__(engine_path, num_buffer)
        self.rgb_means = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
        self.rgb_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255

    def preprocess(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        for i in range(len(inputs)):
            inputs[i] = cv2.resize(inputs[i], (224, 224))
            inputs[i] = cv2.cvtColor(inputs[i], cv2.COLOR_BGR2RGB)
            inputs[i] = (inputs[i] - self.rgb_means) / self.rgb_stds
            inputs[i] = np.transpose(inputs[i], (2, 0, 1))[None]
        return inputs

    def postprocess(self, outputs: List[np.ndarray]) -> Any:
        idxs = np.argmax(outputs[0], axis=1)
        return idxs
