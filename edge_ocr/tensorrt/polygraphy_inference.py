import os
import json
import time
import atexit
import numpy as np
from typing import Generator, Iterator, Union

from cuda import cudart
import tensorrt as trt
from packaging.version import Version

from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

from . import tensorrt_utils as trt_utils
from ..utils.image_utils import load_detect_image, decode_box, load_ocr_image
from ..utils.ocr_utils import ctc_decode, text_add_dash
from ..utils.capture import capture_video
from ..utils.base import BaseInference, UserConfig, InferConfig


trt_version = trt.__version__


class TrtPolygraphyInference(BaseInference):
    def __init__(self, engine_filepath: str, device: int = 0):
        # set_device_success = cudart.cudaSetDevice(device)
        self.device = device
        self.engine = self._load_engine(engine_filepath)
        self.runner = TrtRunner(self.engine)
        self.runner.activate()
        atexit.register(self.runner.deactivate)

        # dummy run
        input_name = self.engine.get_tensor_name(0)
        input_shape = self.engine.get_tensor_shape(input_name)
        _ = self.run(np.random.randn(*input_shape), to_flatten=True)

    def _load_engine(self, engine_filepath: str, severity_value: int = 2):
        return EngineFromBytes(BytesFromPath(engine_filepath))()

    def run(self, input_data: np.ndarray, to_flatten: bool = False):
        input_data = input_data.astype(np.float32)
        res = self.runner.infer({self.engine.get_tensor_name(0): input_data})
        return list(res.values())# OrderedDict


def trt_polygraphy_inference(
        imgsrc: Union[Generator, Iterator],
        user_config: UserConfig,
        config: InferConfig = InferConfig(),
        ):
    LABEL2CHAR = {i + 1: char for i, char in enumerate(config.chars)}

    st = time.time()
    detect_infer = TrtPolygraphyInference(user_config.detect_model_path, device=config.device)
    nms_infer = TrtPolygraphyInference(user_config.nms_model_path, device=config.device)
    crnn_infer = TrtPolygraphyInference(user_config.crnn_model_path, device=config.device)
    print('load engines:', time.time() - st)

    for img in imgsrc:
        # data preprocess
        input_data, img_orig_shape, img_shape = load_detect_image(img)

        detect_res = detect_infer.run(input_data, to_flatten=True)
        bbox_pred = nms_infer.run(detect_res[0][0], to_flatten=False)  # the output should be decoded (rescale to original shape)
        bbox_pred = bbox_pred[0].reshape(10, 6)

        valid_box = np.where(bbox_pred[:, 4] > config.bbox_threshold)[0]
        rescale_output = decode_box(bbox_pred[valid_box, :], img_shape, img_orig_shape)

        if len(rescale_output) <= 0:
            yield img, None, ''
        else:
            bbox = rescale_output[0, :4]
            input_data_ocr = load_ocr_image(img, bbox, extend_ratio=1.15)
            ocr_pred = crnn_infer.run(input_data_ocr, to_flatten=True)
            ocr_pred = ocr_pred[0].reshape(32, 1, 37)

            preds = ctc_decode(ocr_pred, beam_size=10, label2char=LABEL2CHAR)
            text = ''.join(preds[0])
            text = text_add_dash(text)
            if len(text) <= 3:
                text = ''
            text = text.upper()
            yield img, bbox.astype(np.int64), text

