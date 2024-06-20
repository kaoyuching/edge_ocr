import os
import json

from edge_ocr.openvino.openvino_inference import openvino_inference
from edge_ocr.utils.base import UserConfig


if __name__ == '__main__':
    user_config = UserConfig('openvino.env')
    imgsrc = [os.path.join(user_config.data_dir, f) for f in os.listdir(user_config.data_dir)]

    text_results = []
    for path, bbox, res in openvino_inference(imgsrc, user_config):
        print(path, bbox, res)
        text_results.append(res)

    # with open('./results/test_result_yolov5m_53_openvino.json', 'w') as f:
        # res = {'filename': imgsrc, 'text': text_results}
        # json.dump(res, f)
