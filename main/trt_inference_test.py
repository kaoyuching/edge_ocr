import os
from edge_ocr.tensorrt.trt_inference import trt_inference
from edge_ocr.utils.base import UserConfig


if __name__ == '__main__':
    user_config = UserConfig('trt.env')
    imgsrc = [os.path.join("../test_data", f) for f in os.listdir("../test_data")]

    text_results = []
    for res in trt_inference(imgsrc, user_config):
        print(res)
        text_results.append(res)

    # with open('../results/test_result_yolov5m_orin_trt.json', 'w') as f:
        # res = {'filename': imgsrc, 'text': text_results}
        # json.dump(res, f)
