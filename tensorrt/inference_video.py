from trt_inference import trt_inference
from utils.capture import capture_video
from utils.base import UserConfig


if __name__ == '__main__':
    user_config = UserConfig()
    imgsrc = capture_video(video_id=0, inference_rate=1)
    for res in trt_inference(imgsrc, user_config):
        print(res)
