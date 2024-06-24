import cv2

try:
    from edge_ocr.backends import onnx_inference
except ImportError:
    pass
try:
    from edge_ocr.backends import openvino_inference
except ImportError:
    pass
try:
    from edge_ocr.backends import tensorrt_inference
except ImportError:
    pass
try:
    from edge_ocr.backends import polygraphy_inference
except ImportError:
    pass

from edge_ocr.ocr import YoloV5OCR
from edge_ocr.utils.capture import capture_video
from edge_ocr.utils.image_utils import add_bbox_to_frame
from edge_ocr.configs import UserConfig


def main(user_config: UserConfig):
    assert user_config.data.src == 'video'

    imgsrc = capture_video(
        video_id=user_config.data.video_id,
        inference_rate=user_config.data.inference_rate,
        display=False,
    )

    ocr = YoloV5OCR(
        detect_model=user_config.detect_model.get_backend(),
        nms_model=user_config.nms_model.get_backend(),
        crnn_model=user_config.crnn_model.get_backend(),
    )
    for frame, bbox, res in ocr.stream(imgsrc):
        if bbox is not None:
            frame = add_bbox_to_frame(frame, bbox, extend_ratio=1.15)
        cv2.imshow('frame', frame)
        print(bbox, res)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--config-file', default='.env')
    args = parser.parse_args()

    config_file = args.config_file
    if os.path.splitext(config_file)[-1] in ['.ini', '.toml', '.json', '.yml']:
        UserConfig.model_config['config_file'] = config_file
    else:
        UserConfig.model_config['env_file'] = config_file
    UserConfig.model_config['config_file'] = config_file
    user_config = UserConfig()

    main(user_config)
