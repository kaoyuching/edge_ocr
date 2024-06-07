from edge_ocr.onnx.onnx_inference import onnx_inference
from edge_ocr.utils.capture import capture_video
from edge_ocr.utils.base import UserConfig
from edge_ocr.utils.image_utils import extend_box


if __name__ == '__main__':
    user_config = UserConfig('onnx.env')
    imgsrc = capture_video(video_id=0, inference_rate=1, display=False)
    import cv2
    for img, bbox, res in onnx_inference(imgsrc, user_config):
        frame = img.copy()
        if bbox is not None:
            h, w, _ = frame.shape
            bbox = [int(x) for x in extend_box(bbox, ratio=1.15, w=w, h=h)]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        print(bbox, res)
