import os
from edge_ocr.tensorrt.trt_inference import trt_inference
from edge_ocr.utils.base import UserConfig, InferConfig


if __name__ == '__main__':
    user_config = UserConfig('trt.env')
    infer_config = InferConfig(device=1)
    img_dir = "/data/data_set/doriskao/ocr_dataset/car_plate_20230325"
    imgsrc = [os.path.join(img_dir, f) for f in os.listdir(img_dir)][:10]

    text_results = []
    for path, bbox, res in trt_inference(imgsrc, user_config, config=infer_config):
        print(path, bbox, res)
        text_results.append(res)

    # with open('../results/test_result_yolov5m_orin_trt.json', 'w') as f:
        # res = {'filename': imgsrc, 'text': text_results}
        # json.dump(res, f)
