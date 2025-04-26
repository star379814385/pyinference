from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
import tensorrt as trt
from cuda import cudart


class BaseInferece(ABC):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.load_model(model_path)

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        return inputs

    @abstractmethod
    def postprocess(self, outputs: List[np.ndarray]) -> List[np.ndarray]:
        return outputs

    @abstractmethod
    def _infer(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        return inputs

    def infer(self, inputs: List[np.ndarray]) -> Any:
        inputs = self.preprocess(inputs)
        outputs = self._infer(inputs, check_data=False)
        return self.postprocess(outputs)
