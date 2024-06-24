import os
from typing import Optional, Literal

from pydantic import BaseModel, validator


class BaseDataSourceConfig(BaseModel):
    src: str


class VideoDataSource(BaseDataSourceConfig):
    src: Literal['video'] = 'video'
    video_id: int = 0
    """ index of video source """
    inference_rate: Optional[int] = None
    """ frames to inference per second (default: from video FPS) """


class ImagesDataSource(BaseDataSourceConfig):
    src: Literal['images'] = 'images'
    folder: str
    output: Optional[str] = None
    
    @validator('folder')
    def validate_folder(cls, value):
        assert os.path.isdir(value), f'{value} is not a directory'
        return value

    @validator('output')
    def validate_output(cls, value):
        return value or None


DataSourceConfigs = (
    ImagesDataSource,
    VideoDataSource,
)
