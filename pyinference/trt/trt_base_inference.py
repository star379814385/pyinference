from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
import tensorrt as trt
from cuda import cudart

from ..base_inference import BaseInferece
from .trt_common import (
    cuda_call,
    memcpy_device_to_host,
    memcpy_host_to_device,
    do_inference,
    allocate_buffers,
)

"""
Tensorrt Use For Ver 10.9
"""


# class TensorRTInfer(BaseInferece):
#     def __init__(self, model_path: str):
#         super().__init__(model_path)

#     def load_model(self, engine_path: str) -> None:
#         # Load TRT engine
#         self.logger = trt.Logger(trt.Logger.ERROR)
#         with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
#             assert runtime
#             self.engine = runtime.deserialize_cuda_engine(f.read())
#         assert self.engine
#         self.context = self.engine.create_execution_context()
#         assert self.context

#         self.stream = cudart.cudaStreamCreate()

#         # Setup I/O bindings
#         self.inputs = []
#         self.outputs = []
#         self.allocations = []
#         for i in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(i)
#             is_input = False
#             if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
#                 is_input = True
#             dtype = self.engine.get_tensor_dtype(name)
#             shape = self.engine.get_tensor_shape(name)
#             if is_input:
#                 self.batch_size = shape[0]
#             size = np.dtype(trt.nptype(dtype)).itemsize
#             for s in shape:
#                 size *= s
#             allocation = cuda_call(cudart.cudaMalloc(size))
#             binding = {
#                 "index": i,
#                 "name": name,
#                 "dtype": np.dtype(trt.nptype(dtype)),
#                 "shape": list(shape),
#                 "allocation": allocation,
#             }
#             self.allocations.append(allocation)
#             if is_input:
#                 self.inputs.append(binding)
#             else:
#                 self.outputs.append(binding)

#         for i in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(i)
#             is_input = False
#             if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
#                 is_input = True
#             dtype = self.engine.get_tensor_dtype(name)
#             shape = self.engine.get_tensor_shape(name)
#             if is_input:
#                 self.batch_size = shape[0]
#             size = np.dtype(trt.nptype(dtype)).itemsize
#             for s in shape:
#                 size *= s
#             allocation = cuda_call(cudart.cudaMalloc(size))
#             binding = {
#                 "index": i,
#                 "name": name,
#                 "dtype": np.dtype(trt.nptype(dtype)),
#                 "shape": list(shape),
#                 "allocation": allocation,
#             }
#             self.allocations.append(allocation)
#             if is_input:
#                 self.inputs.append(binding)
#             else:
#                 self.outputs.append(binding)

#         assert self.batch_size > 0
#         assert len(self.inputs) > 0
#         assert len(self.outputs) > 0
#         assert len(self.allocations) > 0

#     def inputs_spec(self):
#         """
#         Get the specs for the input tensor of the network. Useful to prepare memory allocations.
#         :return: Two items, the shape of the input tensor and its (numpy) datatype.
#         """
#         # return self.inputs[0]["shape"], self.inputs[0]["dtype"]
#         return [(inp["shape"], inp["dtype"]) for inp in self.inputs]

#     def outputs_spec(self):
#         """
#         Get the specs for the output tensor of the network. Useful to prepare memory allocations.
#         :return: Two items, the shape of the output tensor and its (numpy) datatype.
#         """
#         # return self.outputs[0]["shape"], self.outputs[0]["dtype"]
#         return [(oup["shape"], oup["dtype"]) for oup in self.outputs]

#     def _infer(self, inputs: List[np.ndarray], check_data=True) -> List[np.ndarray]:
#         outputs = [np.zeros(*oup_spec) for oup_spec in self.outputs_spec()]
#         # check data
#         if check_data:
#             assert len(inputs) == len(self.inputs)
#             for inp, inp_info in zip(inputs, self.inputs):
#                 assert inp_info["shape"] == inp.shape
#                 assert inp_info["dtype"] == inp.dtype

#         # # Process I/O and execute the network
#         # for inp, inp_info in zip(inputs, self.inputs):
#         #     memcpy_host_to_device(inp_info["allocation"], np.ascontiguousarray(inp))
#         # self.context.execute_v2(self.allocations)
#         # for oup, oup_info in zip(outputs, self.outputs):
#         #     memcpy_device_to_host(oup, oup_info["allocation"])

#         do_inference(self.context, self.engine, self.allocations, inputs, outputs, self.stream)

#         return outputs

#     def infer(self, inputs: List[np.ndarray]) -> Any:
#         inputs = self.preprocess(inputs)
#         outputs = self._infer(inputs, check_data=False)
#         return self.postprocess(outputs)


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

        self.inputs_info = []
        self.outputs_info = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
            }
            if is_input:
                self.inputs_info.append(binding)
            else:
                self.outputs_info.append(binding)
        self.buffers = []
        for i in range(1):
            inputs, outputs, bindings, stream = allocate_buffers(self.engine, None)
            self.buffers.append(
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "bindings": bindings,
                    "stream": stream,
                }
            )

    def inputs_spec(self):
        return [(inp["shape"], inp["dtype"]) for inp in self.inputs_info]

    def outputs_spec(self):
        return [(oup["shape"], oup["dtype"]) for oup in self.outputs_info]

    def _infer(self, inputs: List[np.ndarray], check_data=True) -> List[np.ndarray]:
        # outputs = [np.zeros(*oup_spec) for oup_spec in self.outputs_spec()]
        # # check data
        if check_data:
            assert len(inputs) == len(self.inputs)
            for inp, inp_info in zip(inputs, self.inputs):
                assert inp_info["shape"] == inp.shape
                assert inp_info["dtype"] == inp.dtype
        buffer = self.buffers[0]
        for i in range(len(inputs)):
            buffer["inputs"][i].host = inputs[i]

        outputs = do_inference(self.context, self.engine, **buffer)
        for i in range(len(outputs)):
            outputs[i] = outputs[i].reshape(self.outputs_info[i]["shape"])

        return outputs

    def infer(self, inputs: List[np.ndarray]) -> Any:
        inputs = self.preprocess(inputs)
        outputs = self._infer(inputs, check_data=False)
        return self.postprocess(outputs)

    def infer_async(self, inputs: List[np.ndarray]) -> Any:
        buffer = None
        for buffer_ in self.buffers:
            event = buffer_.get("event", None)
            if event is None:
                buffer = buffer_
        if buffer is None:
            return False
        end_event = cuda_call(cudart.cudaEventCreate())

        inputs_process = self.preprocess(inputs)
        inputs = buffer["inputs"]
        outputs = buffer["outputs"]
        stream = buffer["stream"]
        bindings = buffer["bindings"]
        for i in range(len(inputs)):
            inputs[i].host = inputs_process[i]

        num_io = self.engine.num_io_tensors
        for i in range(num_io):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), bindings[i])

        kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [
            cuda_call(
                cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)
            )
            for inp in inputs
        ]

        # Run inference.
        self.context.execute_async_v3(stream_handle=stream)
        cuda_call(cudart.cudaEventRecord(end_event, stream))

        # Transfer predictions back from the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [
            cuda_call(
                cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)
            )
            for out in outputs
        ]
        cuda_call(cudart.cudaEventRecord(end_event, stream))
        buffer["event"] = end_event
        return True

    def get_results(self, wait=False):
        results = []
        if wait:
            for buffer in self.buffers:
                cudart.cudaStreamSynchronize(buffer["stream"])
        for buffer in self.buffers:
            event = buffer.get("event", None)
            if event is None:
                continue
            if cudart.cudaEventQuery(event)[0] != cudart.cudaError_t.cudaSuccess:
                continue
            buffer.pop("event")
            outputs = buffer["outputs"]
            res = [
                out.host.reshape(self.outputs_info[i]["shape"])
                for i, out in enumerate(outputs)
            ]
            res = self.postprocess(res)
            results.append(res)
        return results
