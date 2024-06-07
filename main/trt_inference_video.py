import cv2
from edge_ocr.tensorrt.trt_inference import trt_inference
from edge_ocr.utils.capture import capture_video
from edge_ocr.utils.base import UserConfig
from edge_ocr.utils.image_utils import add_bbox_to_frame


if __name__ == '__main__':
    user_config = UserConfig('trt.env')
    imgsrc = capture_video(video_id=0, inference_rate=30, display=False)
    for frame, bbox, res in trt_inference(imgsrc, user_config):
        if bbox is not None:
            frame = add_bbox_to_frame(frame, bbox, extend_ratio=1.15)
        cv2.imshow('frame', frame)
        print(bbox, res)
