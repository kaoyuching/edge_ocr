from typing import Union

from pydantic import Field
from pydantic_config import SettingsModel, SettingsConfig

from .backends import BackendConfigs
from .datasources import DataSourceConfigs
from .ocr import InferConfig


class UserConfig(SettingsModel):
    detect_model: Union[tuple(BackendConfigs)] = Field(..., discriminator='backend')
    nms_model: Union[tuple(BackendConfigs)] = Field(..., discriminator='backend')
    crnn_model: Union[tuple(BackendConfigs)] = Field(..., discriminator='backend')

    infer_config: InferConfig = InferConfig()
    data: Union[tuple(DataSourceConfigs)] = Field(..., discriminator='src')

    model_config = SettingsConfig(
        extra='ignore',
        case_insensitive=False,
        from_attributes=True,
        env_nested_delimiter='__',
    )
