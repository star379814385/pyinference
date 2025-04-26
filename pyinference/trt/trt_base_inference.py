from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
import tensorrt as trt
from cuda import cudart

from ..base_inference import BaseInferece
from .trt_common import cuda_call, memcpy_device_to_host, memcpy_host_to_device

"""
Tensorrt Use For Ver 10.9
"""


class TensorRTInfer(BaseInferece):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def load_model(self, engine_path: str) -> None:
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def inputs_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        # return self.inputs[0]["shape"], self.inputs[0]["dtype"]
        return [(inp["shape"], inp["dtype"]) for inp in self.inputs]

    def outputs_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        # return self.outputs[0]["shape"], self.outputs[0]["dtype"]
        return [(oup["shape"], oup["dtype"]) for oup in self.outputs]

    def _infer(self, inputs: List[np.ndarray], check_data=True) -> List[np.ndarray]:
        outputs = [np.zeros(*oup_spec) for oup_spec in self.outputs_spec()]
        # check data
        if check_data:
            assert len(inputs) == len(self.inputs)
            for inp, inp_info in zip(inputs, self.inputs):
                assert inp_info["shape"] == inp.shape
                assert inp_info["dtype"] == inp.dtype

        # Process I/O and execute the network
        for inp, inp_info in zip(inputs, self.inputs):
            memcpy_host_to_device(inp_info["allocation"], np.ascontiguousarray(inp))
        self.context.execute_v2(self.allocations)
        for oup, oup_info in zip(outputs, self.outputs):
            memcpy_device_to_host(oup, oup_info["allocation"])
        return outputs

    def infer(self, inputs: List[np.ndarray]) -> Any:
        inputs = self.preprocess(inputs)
        outputs = self._infer(inputs, check_data=False)
        return self.postprocess(outputs)
