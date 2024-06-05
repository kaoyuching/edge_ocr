import argparse
from typing import Tuple, List
import time
import atexit
import numpy as np

import onnx
import tensorrt as trt
from cuda import cuda, cudart

trt_version = trt.__version__

r'''
$ CUDA_MODULE_LOADING='LAZY' python ./onnx_to_tensorrt.py '../onnx_models/yolov5_post_nms_xyxy_single.onnx' 16128 6 --memory_pool_size 2 --save_engine --engine_filepath "./trt_engine/nms.engine"
'''


parser = argparse.ArgumentParser()
parser.add_argument('onnx_model_path', type=str, help='onnx model filepath')
parser.add_argument('input_shape', nargs='+', type=int, help='define input data type')
parser.add_argument('--device', type=int, default=0, help='device number')
parser.add_argument('--memory_pool_size', type=float, default=2)
parser.add_argument('--save_engine', action='store_true')
parser.add_argument('--engine_filepath', type=str, default='./trt_engine.engine')
parser.add_argument('--severity_value', type=int, default=2)
args = parser.parse_args()


onnx_model_path = args.onnx_model_path
input_shape = tuple(args.input_shape)
device = args.device
memory_pool_size = args.memory_pool_size
save_engine = args.save_engine
engine_filepath = args.engine_filepath
severity_value = args.severity_value

print('onnx model:', onnx_model_path)
print('input shape:', input_shape)

# set certain device
set_device_success = cudart.cudaSetDevice(device)
print('set cuda device:', set_device_success)


def build_engine_from_onnx(
    onnx_model_path: str,
    input_shape: Tuple[int],
    memory_pool_size: float = 2,
    save_engine: bool = False,
    engine_filepath: str = './trt_engine.engine',
    severity_value: int = 2,
):
    r'''
        - onnx_model_path (str): the onnx model path.
        - input_shape (tuple): model input data shape.
        - memory_pool_size (float): limit memory pool size in GB. Default: 2 GB.
        - save_engine (bool): whether to save engine or not.
        - engine_filepath (str): the filepath of saved engine.
        - severity_value (int): {0: INTERNAL_ERROR, 1: ERROR, 2: WARNING, 3: INFO, 4: VERBOSE}. Default: WARNING{2}.
    '''
    logger = trt.Logger(trt.Logger.Severity(severity_value))
    builder = trt.Builder(logger)

    # define network
    # trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH must be set when use TensorRT version 8 or 9
    network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_creation_flag)  # INetworkDefinition

    # parse onnx model
    onnx_model = onnx.load(onnx_model_path)
    parser = trt.OnnxParser(network, logger)
    if trt_version >= '8.6':
        parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)  # tensorrt >= v8.6
    parse_success = parser.parse(onnx_model.SerializeToString())
    # parse_success = parser.parse_from_file(onnx_model_path)
    if not parse_success:
        print(parser.get_error(0))

    # set up optimization profile
    profile = builder.create_optimization_profile()
    input_tensor_name = onnx_model.graph.input[0].name
    profile.set_shape(input_tensor_name, input_shape, input_shape, input_shape)  # the name of the input tensor should be the same as the name of the input tensor in onnx model

    # set builder config: tensorrt.tensorrt.IBuilderConfig
    builder_config = builder.create_builder_config()
    pool_type = trt.MemoryPoolType.WORKSPACE
    builder_config.set_memory_pool_limit(pool_type, int(memory_pool_size * 1e9))
    if trt_version >= '9.0':
        # create a version-compatible engine
        builder_config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)  # tensorrt >= version9.0
    builder_config.flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
    # add optimization profile
    is_valid = builder_config.add_optimization_profile(profile)
    if is_valid == -1:
        print('The input is not valid. The index of the optimization profile is -1.')

    # create an engine from a builder
    engine_bytes = builder.build_serialized_network(network, builder_config)  # return IHostMemory
    runtime = trt.Runtime(logger)
    if trt_version >= '9.0':
        # Indicate to TensorRT that you trust the plan
        runtime.engine_host_code_allowed = True  # after version 9.0
    engine = runtime.deserialize_cuda_engine(engine_bytes)  # return ICudaEngine

    # save engine
    if save_engine:
        with open(engine_filepath, 'wb') as f:
            f.write(engine_bytes)
    return engine


engine = build_engine_from_onnx(
    onnx_model_path,
    input_shape,
    memory_pool_size=memory_pool_size,
    save_engine=save_engine,
    engine_filepath=engine_filepath,
    severity_value=severity_value,
)
