import logging
import traceback
from collections import OrderedDict
from typing import List, Dict, Union

import numpy as np
import tensorrt as trt
from packaging.version import Version
from cuda import cudart

from .base import BaseModelInference
from . import tensorrt_utils as trt_utils


logger = logging.getLogger(__name__)


class TensorrtModelInference(BaseModelInference):
    _model_type = 'tensorrt engine'

    def __init__(self, model_path: str, device: int = 0, severity_value: int = 2):
        super().__init__(model_path)
        self.device = device
        self.severity_value = severity_value
        self.engine = None
        self._context = None
        self._inputs = None
        self._outputs = None
        self._bindings = None
        self._stream = None
        self.input_names = None
        self.output_names = None

    def _load_model(self):
        set_device_success = cudart.cudaSetDevice(self.device)
        trt_logger = trt.Logger(trt.Logger.Severity(self.severity_value))
        with open(self.model_path, 'rb') as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(trt_logger)
        if Version(trt.__version__) >= Version('9.0'):
            # Indicate to TensorRT that you trust the plan
            runtime.engine_host_code_allowed = True  # after version 9.0
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)  # return ICudaEngine
        self._context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        self.input_metadata = {}
        self.output_metadata = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            metadata = {
                'name': name,
                'shape': self.engine.get_tensor_shape(name),
                'dtype': self.engine.get_tensor_dtype(name),
            }
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_metadata[name] = metadata
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                self.output_metadata[name] = metadata

    def _activate(self):
        self._context = self.engine.create_execution_context()
        (
            self._inputs,
            self._outputs,
            self._bindings,
            self._stream,
        ) = trt_utils.allocate_buffers(self.engine)
        # dummy run
        try:
            dummy_inputs = {
                k: np.zeros(metadata['shape'], dtype=trt.nptype(metadata['dtype']))
                for k, metadata in self.input_metadata.items()
            }
        except Exception as e:
            logger.warn('Fail to run dummy inputs, skip.')
            logger.debug(traceback.format_exc())
            traceback.print_exc()

    def _deactivate(self):
        trt_utils.free_buffers(self._inputs, self._outputs, self._stream)

    def _run(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        set_device_success = cudart.cudaSetDevice(self.device)
        if isinstance(input_data, dict):
            input_data = [input_data[k] for k in self.input_names]
        for i, data in enumerate(input_data):
            # flatten and copy to buffer
            np.copyto(self._inputs[i].host, data.ravel())
        outputs = trt_utils.do_inference(
            self._context,
            engine=self.engine,
            bindings=self._bindings,
            inputs=self._inputs,
            outputs=self._outputs,
            stream=self._stream,
        )
        # collect and unflatten
        outputs = OrderedDict([
            (k, output.reshape(self.output_metadata[k]['shape']))
            for k, output in zip(self.output_names, outputs)
        ])
        return outputs
