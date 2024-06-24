import cv2

from edge_ocr.backends import (
    onnx_inference,
    openvino_inference,
    tensorrt_inference,
    polygraphy_inference,
)
from edge_ocr.configs import UserConfig
from edge_ocr.ocr import YoloV5OCR
from edge_ocr.utils.capture import capture_video
from edge_ocr.utils.image_utils import add_bbox_to_frame


def main(config_file: str = '.env'):
    UserConfig.model_config['config_file'] = config_file
    user_config = UserConfig()

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
    main()
