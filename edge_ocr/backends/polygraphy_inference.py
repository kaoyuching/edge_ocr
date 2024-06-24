import logging
import traceback
from collections import OrderedDict
from typing import List, Dict, Union

import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

from .base import BaseModelInference


logger = logging.getLogger(__name__)


class PolygraphyModelInference(BaseModelInference):
    _model_type = 'tensorrt engine'

    def __init__(self, model_path: str, device: int = -1):
        super().__init__(model_path)
        self.device = device
        self.engine = None
        self.engine_runner = None

    def _load_model(self):
        # TODO: device
        self.engine = EngineFromBytes(BytesFromPath(self.model_path))()
        self.engine_runner = TrtRunner(self.engine)

    def _activate(self):
        self.engine_runner.activate()
        # dummy run
        try:
            input_metadata = self.engine_runner.get_input_metadata()
            dummy_inputs = {
                k: np.zeros(metadata.shape, dtype=metadata.dtype)
                for k, metadata in input_metadata.items()
            }
        except Exception as e:
            logger.warn('Fail to run dummy inputs, skip.')
            logger.debug(traceback.format_exc())

    def _deactivate(self):
        self.engine_runner.deactivate()

    def _run(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        if isinstance(input_data, list):
            input_data = {self.engine.get_tensor_name(i): data for i, data in enumerate(input_data)}
        return self.engine_runner.infer(input_data)
