import os
import json
from typing import List, Union, Generator, Iterator
import numpy as np
import time
# import atexit
import onnx
import onnxruntime

from ..utils.image_utils import load_detect_image, decode_box, load_ocr_image
from ..utils.ocr_utils import ctc_decode, text_add_dash
from ..utils.base import BaseInference, UserConfig, InferConfig


class OnnxInference(BaseInference):
    def __init__(self, engine_filepath: str, device: int = -1):
        self.device = device
        self.session = self._load_engine(engine_filepath)

    def _load_engine(self, engine_filepath: str):
        assert os.path.exists(engine_filepath)
        if self.device >= 0:
            providers = [
               ('CUDAExecutionProvider', {'device_id': self.device},), 
               'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']

        # disable the memory pattern optimization to avoid getting wrong result after second run
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_mem_pattern = False
        session = onnxruntime.InferenceSession(
            engine_filepath,
            sess_options=sess_options,
            providers=providers,
        )
        return session

    def run(self, input_data: List[np.ndarray]):
        if self.device == -1:
            outputs = self._run_cpu(input_data)
        else:
            outputs = self._run_gpu(input_data)
        return outputs

    def _run_cpu(self, input_data: List[np.ndarray]):
        if not isinstance(input_data, list):
            input_data = [input_data]
        session_inputs = self.session.get_inputs()
        data = {
            sess.name: input_data[i].astype(np.float32)
            for i, sess in enumerate(session_inputs)
        }
        outputs = self.session.run(None, data)
        return outputs

    def _run_gpu(self, input_data: List[np.ndarray]):
        if not isinstance(input_data, list):
            input_data = [input_data]
        input_data_ort = [onnxruntime.OrtValue.ortvalue_from_numpy(x, 'cuda', self.device) for x in input_data]
        session_inputs = self.session.get_inputs()
        io_binding = self.session.io_binding()
        for i, sess in enumerate(session_inputs):
            # io_binding.bind_cpu_input(sess.name, input_data[i])
            io_binding.bind_input(
                name=sess.name, 
                device_type=input_data_ort[i].device_name(),
                device_id=self.device,
                element_type=np.float32,
                shape=input_data_ort[i].shape(),
                buffer_ptr=input_data_ort[i].data_ptr()
            )
        io_binding.bind_output('output', device_type='cuda', device_id=self.device)
        self.session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()
        return outputs


def onnx_inference(
        imgsrc: Union[Generator, Iterator],
        user_config: UserConfig,
        config: InferConfig = InferConfig(),
        ):
    LABEL2CHAR = {i + 1: char for i, char in enumerate(config.chars)}

    st = time.time()
    device = -1 # CPU
    detect_infer = OnnxInference(user_config.detect_model_path, device=device)
    print('load yolov5')
    nms_infer = OnnxInference(user_config.nms_model_path, device=device)
    print('load nms')
    crnn_infer = OnnxInference(user_config.crnn_model_path, device=device)
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
