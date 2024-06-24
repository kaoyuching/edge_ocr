import os
from collections import OrderedDict
from typing import List, Dict, Union

import numpy as np
from openvino.runtime import Core

from .base import BaseModelInference


class OpenvinoModelInference(BaseModelInference):
    _model_type = 'openvino compiled model'

    def __init__(self, model_xml_path: str, device: str = 'CPU'):
        super().__init__(model_xml_path)
        self.device = device
        self.model = None
        self.compiled_model = None
        # core.set_property('CPU', {'INFERENCE_NUM_THREADS': 5})
        # available_devices = core.available_devices
        # print('Available devices:', available_devices)

    def _load_model(self):
        core = Core()
        self.model = core.read_model(model=self.model_path)
        self.compiled_model = core.compile_model(self.model, device_name=self.device)
        # print(compiled_model.input(0).partial_shape)
        # print(compiled_model.input(0).partial_shape[1].get_length())
        # print(compiled_model.output(0).partial_shape)

    def _run(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        return self.compiled_model(input_data)
