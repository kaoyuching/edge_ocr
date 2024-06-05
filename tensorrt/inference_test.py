import os
from trt_inference import trt_inference
from utils.base import UserConfig


if __name__ == '__main__':
    user_config = UserConfig('.env')
    imgsrc = [os.path.join("../../test_data", f) for f in os.listdir("../../test_data")]
    for res in trt_inference(imgsrc, user_config):
        print(res)
