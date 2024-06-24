import abc
import time
from collections import OrderedDict
from typing import Any, List, Dict, Optional, Union

import numpy as np


class BaseModelInference(abc.ABC):
    _is_active: bool = False
    _is_loaded: bool = False
    _model_type: Optional[str] = None
    
    def __init__(self, model_path: str):
        self.model_path = model_path

    @property
    def model_type(self):
        return self._model_type or self.__class__.__name__

    def load_model(self):
        if not self._is_loaded:
            self._load_model()
            self._is_loaded = True

    @abc.abstractmethod
    def _load_model(self):
        raise NotImplementedError()

    def activate(self):
        if not self._is_loaded:
            raise ValueError('model is not loaded, please call load_model() first')
        if not self._is_active:
            self._activate()
            self._is_active = True

    def _activate(self):
        pass

    def deactivate(self):
        if self._is_active:
            self._deactivate()
            self._is_active = False

    def _deactivate(self):
        pass

    def run(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        if not self._is_loaded:
            raise ValueError('model is not loaded, please call load_model() first')
        if not self._is_active:
            raise ValueError('model is not active, please call activate() first')
        return self._run(input_data)

    @abc.abstractmethod
    def _run(self,
            input_data: Union[Dict[str, np.ndarray], List[np.ndarray]],
            ) -> 'OrderedDict[str, np.ndarray]':
        raise NotImplementedError()


class MultiModelInference(abc.ABC):
    def __init__(self, models: Union[Dict[Any, BaseModelInference], List[BaseModelInference]]):
        self.models = models

    @property
    def _models(self):
        return self.models.values() if isinstance(self.models, dict) else self.models

    def load_models(self):
        st = time.time()
        if isinstance(self.models, dict):
            it = self.models.items()
        else:
            it = enumerate(self.models)
        for k, m in it:
            print(f'Load {m.model_type}: {k}')
            m.load_model()
        dt = time.time() - st
        print(f'load models cost {dt}s')

    def activate(self):
        for m in self._models:
            m.activate()

    def deactivate(self):
        for m in self._models:
            m.deactivate()

    @abc.abstractmethod
    def run(self, input_data: Any) -> Any:
        raise NotImplementedError()
