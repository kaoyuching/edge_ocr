from typing import Type
from .base import BaseBackendConfig, BaseInferenceBackend

__all__ = ['BackendConfigs', 'BaseBackendConfig', 'BaseInferenceBackend']

BackendConfigs = []


def register_backend(
        backend_cls: Type[BaseInferenceBackend],
        config_schema: Type[BaseBackendConfig],
        ):
    BackendConfigs.append(config_schema)
    g = globals()
    cls_name = backend_cls.__name__
    g[cls_name] = backend_cls
    __all__.append(cls_name)
