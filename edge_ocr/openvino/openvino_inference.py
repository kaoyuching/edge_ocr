import os
import time

from typing import List, Union, Generator, Iterator

import numpy as np
from openvino.runtime import Core

from ..utils.image_utils import load_detect_image, decode_box, load_ocr_image
from ..utils.ocr_utils import ctc_decode, text_add_dash
from ..utils.base import BaseInference, UserConfig, InferConfig


class OpenvinoInference(BaseInference):
    def __init__(self, model_xml_path: str, device: str = 'CPU'):
        self.device = device
        self.compiled_model = self._load_engine(model_xml_path)

    def _load_engine(self, model_xml_path: str):
        assert os.path.exists(model_xml_path)
        core = Core()
        # core.set_property('CPU', {'INFERENCE_NUM_THREADS': 5})
        available_devices = core.available_devices
        print('Available devices:', available_devices)
        model = core.read_model(model=model_xml_path)
        compiled_model = core.compile_model(model, device_name=self.device)
        # print(compiled_model.input(0).partial_shape)
        # print(compiled_model.input(0).partial_shape[1].get_length())
        # print(compiled_model.output(0).partial_shape)
        return compiled_model

    def run(self, input_data: List[np.ndarray]):
        outputs = self.compiled_model(input_data)
        return [outputs[output_layer] for output_layer in self.compiled_model.outputs]


def openvino_inference(
        imgsrc: Union[Generator, Iterator],
        user_config: UserConfig,
        config: InferConfig = InferConfig(),
        ):
    LABEL2CHAR = {i + 1: char for i, char in enumerate(config.chars)}

    st = time.time()
    device = 'CPU'
    detect_infer = OpenvinoInference(user_config.detect_model_path, device=device)
    print('load yolov5')
    nms_infer = OpenvinoInference(user_config.nms_model_path, device=device)
    print('load nms')
    crnn_infer = OpenvinoInference(user_config.crnn_model_path, device=device)
    print('load sessions:', time.time() - st)

    for img in imgsrc:
        # data preprocess
        input_data, img_orig_shape, img_shape = load_detect_image(img)

        detect_res = detect_infer.run(input_data)  # shape: (1, 16128, 6) / effdetd0: (1, 9, 64, 64)
        bbox_pred = nms_infer.run(detect_res[0][0])  # the output should be decoded (rescale to original shape)
        bbox_pred = bbox_pred[0].reshape(10, 6)

        valid_box = np.where(bbox_pred[:, 4] > config.bbox_threshold)[0]
        rescale_output = decode_box(bbox_pred[valid_box, :], img_shape, img_orig_shape)

        if len(rescale_output) <= 0:
            yield img, None, ''
        else:
            bbox = rescale_output[0, :4] # xyxy
            input_data_ocr = load_ocr_image(img, bbox, extend_ratio=1.15)
            ocr_pred = crnn_infer.run(input_data_ocr)
            ocr_pred = ocr_pred[0]  # shape (32, 1, 37)

            preds = ctc_decode(ocr_pred, beam_size=10, label2char=LABEL2CHAR)
            text = ''.join(preds[0])
            text = text_add_dash(text)
            if len(text) <= 3:
                text = ''
            text = text.upper()
            yield img, bbox.astype(np.int64), text
