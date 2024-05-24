import os
import numpy as np
import time
import tensorrt as trt
import atexit
from cuda import cudart

import tensorrt_utils as trt_utils
from utils.image_utils import load_detect_image, decode_box, load_ocr_image
from utils.ocr_utils import ctc_decode


trt_version = trt.__version__


class TrtInference:
    def __init__(self, engine_filepath: str, device: int = 0):
        set_device_success = cudart.cudaSetDevice(device)
        self.engine = self._load_engine(engine_filepath)
        print('engine:', self.engine)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = trt_utils.allocate_buffers(self.engine)
        atexit.register(trt_utils.free_buffers, self.inputs, self.outputs, self.stream)

        # dummy run
        input_name = self.engine.get_tensor_name(0)
        input_shape = self.engine.get_tensor_shape(input_name)
        _ = self.run(np.random.randn(*input_shape), to_flatten=True)

    def _load_engine(self, engine_filepath: str, severity_value: int = 2):
        logger = trt.Logger(trt.Logger.Severity(severity_value))
        # load engine bytes
        assert os.path.exists(engine_filepath)
        with open(engine_filepath, 'rb') as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(logger)
        if trt_version >= '9.0':
            # Indicate to TensorRT that you trust the plan
            runtime.engine_host_code_allowed = True  # after version 9.0
        engine = runtime.deserialize_cuda_engine(engine_bytes)  # return ICudaEngine
        return engine

    def run(self, input_data: np.ndarray, to_flatten: bool = False):
        if to_flatten:
            input_data = input_data.ravel()
        np.copyto(self.inputs[0].host, input_data)
        trt_outputs = trt_utils.do_inference(
            self.context,
            engine=self.engine,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
        )
        return trt_outputs


CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
LABEL2CHAR = {i + 1: char for i, char in enumerate(CHARS)}

st = time.time()
yolov5_infer = TrtInference('/home/doriskao/workspace/test_edge/tensorrt/trt_engine/yolov5s.engine', device=1)
nms_infer = TrtInference('/home/doriskao/workspace/test_edge/tensorrt/trt_engine/nms.engine', device=1)
crnn_infer = TrtInference('/home/doriskao/workspace/test_edge/tensorrt/trt_engine/crnn.engine', device=1)
print('load engines:', time.time() - st)

print('Test...')
for i in range(1):
    st = time.time()
    img_path = '/data/data_set/doriskao/ocr_dataset/car_plate_20230325/011a65e23a51dd846c0e698d8cf6e52a13431970.png'
    # data preprocess
    input_data, img_orig_shape, img_shape = load_detect_image(img_path)
    print('load data:', time.time() - st)

    yolo_pred = yolov5_infer.run(input_data, to_flatten=True)
    bbox_pred = nms_infer.run(yolo_pred[0], to_flatten=False)  # the output should be decoded (rescale to original shape)
    bbox_pred = bbox_pred[0].reshape(10, 6)

    bbox_threshold = 0.15
    valid_box = np.where(bbox_pred[:, 4] > bbox_threshold)[0]
    rescale_output = decode_box(bbox_pred[valid_box, :], img_shape, img_orig_shape)

    bbox = rescale_output[0, :4]
    input_data_ocr = load_ocr_image(img_path, bbox, extend_ratio=1.15)
    ocr_pred = crnn_infer.run(input_data_ocr, to_flatten=True)
    ocr_pred = ocr_pred[0].reshape(32, 1, 37)

    preds = ctc_decode(ocr_pred, beam_size=10, label2char=LABEL2CHAR)
    text = ''.join(preds[0])

    print('text:', text)
    print('infer time:', time.time() - st)


r'''
def load_engine(engine_filepath: str, severity_value: int = 2,):
    logger = trt.Logger(trt.Logger.Severity(severity_value))
    # load engine bytes
    assert os.path.exists(engine_filepath)
    with open(engine_filepath, 'rb') as f:
        engine_bytes = f.read()

    runtime = trt.Runtime(logger)
    if trt_version >= '9.0':
        # Indicate to TensorRT that you trust the plan
        runtime.engine_host_code_allowed = True  # after version 9.0
    engine = runtime.deserialize_cuda_engine(engine_bytes)  # return ICudaEngine
    return engine


def run_engine(engine, input_data: np.ndarray, to_flatten: bool = False):
    context = engine.create_execution_context()  # return IExecutionContext

    import tensorrt_utils as trt_utils
    # allocate all buffers required for an engine. host/device inputs/outputs
    inputs, outputs, bindings, stream = trt_utils.allocate_buffers(engine)

    np.copyto(inputs[0].host, input_data.ravel())
    print('do inference...')
    trt_outputs = trt_utils.do_inference(
        context,
        engine=engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    return trt_outputs


    # @atexit.register
    # def free_buffer():
        # print('free buffers')
        # trt_utils.free_buffers(inputs, outputs, stream)
    # print('result:', pred.shape)


CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
LABEL2CHAR = {i + 1: char for i, char in enumerate(CHARS)}

# set device is important
set_device_success = cudart.cudaSetDevice(1)

yolov5_engine = load_engine('/home/doriskao/workspace/test_edge/tensorrt/trt_engine/yolov5s.engine')
nms_engine = load_engine('/home/doriskao/workspace/test_edge/tensorrt/trt_engine/nms.engine')
crnn_engine = load_engine('/home/doriskao/workspace/test_edge/tensorrt/trt_engine/crnn.engine')


img_path = '/data/data_set/doriskao/ocr_dataset/car_plate_20230325/011a65e23a51dd846c0e698d8cf6e52a13431970.png'
# data preprocess
input_data, img_orig_shape, img_shape = load_detect_image(img_path)
print('input data shape:', input_data.shape)  # (1, 3, 512, 512)
# input_data = np.random.randn(1, 3, 512, 512)

yolo_pred = run_engine(yolov5_engine, input_data)
# yolo_pred = trt_outputs[0].reshape(16128, 6)
bbox_pred = run_engine(nms_engine, yolo_pred[0])  # the output should be decoded (rescale to original shape)
bbox_pred = bbox_pred[0].reshape(10, 6)
print('bbox:', bbox_pred.shape, bbox_pred[0])

bbox_threshold = 0.15
valid_box = np.where(bbox_pred[:, 4] > bbox_threshold)[0]
rescale_output = decode_box(bbox_pred[valid_box, :], img_shape, img_orig_shape)

# input_data_ocr = np.random.randn(1, 3, 64, 512)
bbox = rescale_output[0, :4]
input_data_ocr = load_ocr_image(img_path, bbox, extend_ratio=1.15)
ocr_pred = run_engine(crnn_engine, input_data_ocr)
ocr_pred = ocr_pred[0].reshape(32, 1, 37)

preds = ctc_decode(ocr_pred, beam_size=10, label2char=LABEL2CHAR)
text = ''.join(preds[0])

print('text:', text)
'''
