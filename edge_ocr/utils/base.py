import abc
import numpy as np
from pydantic import BaseSettings


class InferConfig(BaseSettings):
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    bbox_threshold = 0.15
    device = 0

    class Config:
        case_insensitive = False


class UserConfig(BaseSettings):
    detect_model_path: str
    nms_model_path: str
    crnn_model_path: str

    class Config:
        orm_mode=True
        env_file='.env'
        case_insensitive = False


class BaseInference(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self, input_data: np.ndarray, **kwargs):
        raise NotImplementedError()
