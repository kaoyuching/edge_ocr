import sys
sys.path.append('../tensorrt/')

import os
import json
from typing import List
import numpy as np
import time
# import atexit
import onnx
import onnxruntime

from utils.image_utils import load_detect_image, decode_box, load_ocr_image
from utils.ocr_utils import ctc_decode, text_add_dash


class OnnxInference:
    def __init__(self, engine_filepath: str, device: int = 0):
        # set_device_success = cudart.cudaSetDevice(device)
        self.session = self._load_engine(engine_filepath)

    def _load_engine(self, engine_filepath: str):
        assert os.path.exists(engine_filepath)
        session = onnxruntime.InferenceSession(
            engine_filepath,
            # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        return session

    def run(self, input_data: List[np.ndarray]):
        if not isinstance(input_data, list):
            input_data = [input_data]
        session_inputs = self.session.get_inputs()
        data = {
            sess.name: input_data[i].astype(np.float32)
            for i, sess in enumerate(session_inputs)
        }
        outputs = self.session.run(None, data)
        return outputs


if __name__ == '__main__':
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    LABEL2CHAR = {i + 1: char for i, char in enumerate(CHARS)}

    st = time.time()
    yolov5_infer = OnnxInference('/home/doriskao/workspace/test_edge/onnx_models/yolov5s.onnx', device=1)
    print('load yolov5')
    nms_infer = OnnxInference('/home/doriskao/workspace/test_edge/onnx_models/yolov5_post_nms_xyxy_single.onnx', device=1)
    print('load nms')
    crnn_infer = OnnxInference('/home/doriskao/workspace/test_edge/onnx_models/crnn.onnx', device=1)
    print('load sessions:', time.time() - st)

    img_paths = [os.path.join("/data/data_set/doriskao/ocr_dataset/car_plate_20230325", f) for f in os.listdir("/data/data_set/doriskao/ocr_dataset/car_plate_20230325")]
    text_results = []

    print('Test...')
    for img_path in img_paths:
        # img_path = '/data/data_set/doriskao/ocr_dataset/car_plate_20230325/011a65e23a51dd846c0e698d8cf6e52a13431970.png'
        # data preprocess
        input_data, img_orig_shape, img_shape = load_detect_image(img_path)

        yolo_pred = yolov5_infer.run(input_data)  # shape: (1, 16128, 6)
        bbox_pred = nms_infer.run(yolo_pred[0][0])  # the output should be decoded (rescale to original shape)
        bbox_pred = bbox_pred[0].reshape(10, 6)

        bbox_threshold = 0.15
        valid_box = np.where(bbox_pred[:, 4] > bbox_threshold)[0]
        rescale_output = decode_box(bbox_pred[valid_box, :], img_shape, img_orig_shape)

        if len(rescale_output) <= 0:
            text = ''
        else:
            bbox = rescale_output[0, :4]
            input_data_ocr = load_ocr_image(img_path, bbox, extend_ratio=1.15)
            ocr_pred = crnn_infer.run(input_data_ocr)
            ocr_pred = ocr_pred[0]  # shape (32, 1, 37)

            preds = ctc_decode(ocr_pred, beam_size=10, label2char=LABEL2CHAR)
            text = ''.join(preds[0])
            text = text_add_dash(text)
            if len(text) <= 3:
                text = ''
        text = text.upper()
        # print('text:', text)
        text_results.append(text)

    # with open('/home/doriskao/project/ocr/test_result_yolov5m_53trt.json', 'w') as f:
        # res = {'filename': img_paths, 'text': text_results}
        # json.dump(res, f)
