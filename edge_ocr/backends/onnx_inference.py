from collections import OrderedDict
from typing import List, Dict, Union

import numpy as np
import onnx
import onnxruntime

from .base import BaseModelInference


class OnnxModelInference(BaseModelInference):
    _model_type = 'onnx session'

    def __init__(self, model_path: str, device: int = -1):
        super().__init__(model_path)
        self.device = device
        self.session = None

    def _load_model(self):
        if self.device >= 0:
            providers = [
               ('CUDAExecutionProvider', {'device_id': self.device}), 
               'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']

        # disable the memory pattern optimization to avoid getting wrong result after second run
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_mem_pattern = False
        self.session = onnxruntime.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
        )

    def _run(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        if self.device < 0:
            return self._run_cpu(input_data)
        else:
            return self._run_gpu(input_data)

    def _handle_inputs(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        inputs = OrderedDict([
            (node.name, input_data[i] if isinstance(input_data, list) else input_data[node.name])
            for i, node in enumerate(self.session.get_inputs())
        ])
        return inputs

    def _handle_outputs(self,
            output_data: List[np.ndarray],
            ) -> 'OrderedDict[str, np.ndarray]':
        outputs = OrderedDict([
            (node.name, output)
            for node, output in zip(self.session.get_outputs(), output_data)
        ])
        return outputs

    def _run_cpu(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        input_data = self._handle_inputs(input_data)
        outputs = self.session.run(None, input_data)
        return self._handle_outputs(outputs)

    def _run_gpu(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        input_data = self._handle_inputs(input_data)
        if not isinstance(input_data, list):
            input_data = [input_data]
        io_binding = self.session.io_binding()
        for name, data in input_data.items():
            data_ort = onnxruntime.OrtValue.ortvalue_from_numpy(data, 'cuda', self.device)
            io_binding.bind_input(
                name=name, 
                device_type=data_ort.device_name(),
                device_id=self.device,
                element_type=data_ort.element_type(),
                shape=data_ort.shape(),
                buffer_ptr=data_ort.data_ptr(),
            )
        for name in self.session.get_outputs():
            io_binding.bind_output(name, device_type='cuda', device_id=self.device)
        # run
        self.session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()
        return self._handle_outputs(output_data)
