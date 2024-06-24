import os
import json

from edge_ocr.backends import (
    onnx_inference,
    openvino_inference,
    tensorrt_inference,
    polygraphy_inference,
)
from edge_ocr.configs import UserConfig
from edge_ocr.ocr import YoloV5OCR


def main(config_file: str = '.env'):
    UserConfig.model_config['env_file'] = config_file
    user_config = UserConfig()

    assert user_config.data.src == 'images'

    data_dir = user_config.data.folder
    imgsrc = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    ocr = YoloV5OCR(
        detect_model=user_config.detect_model.get_backend(),
        nms_model=user_config.nms_model.get_backend(),
        crnn_model=user_config.crnn_model.get_backend(),
    )
    text_results = []
    for path, bbox, res in ocr.stream(imgsrc[:10]):
        print(path, bbox, res)
        text_results.append(res)

    output_path = user_config.data.output
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f'save result to {output_path}')
        with open(output_path, 'w') as f:
            res = {'filename': imgsrc, 'text': text_results}
            json.dump(res, f)


if __name__ == '__main__':
    main()
