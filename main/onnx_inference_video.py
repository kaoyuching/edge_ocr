from edge_ocr.onnx.onnx_inference import onnx_inference
from edge_ocr.utils.capture import capture_video
from edge_ocr.utils.base import UserConfig


if __name__ == '__main__':
    user_config = UserConfig('onnx.env')
    imgsrc = capture_video(video_id=0, inference_rate=1, display=True)
    for img, bbox, res in onnx_inference(imgsrc, user_config):
        print(bbox, res)
