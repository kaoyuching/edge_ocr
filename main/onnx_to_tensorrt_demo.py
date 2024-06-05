from typing import List
import time
import atexit
import numpy as np

# import onnx
import tensorrt as trt
from cuda import cuda, cudart
# import onnx_tensorrt.backend as backend

r'''
Jetson Orin Nano (4GB)
    - Jetpack 5.1.2: https://developer.nvidia.com/embedded/jetpack-sdk-512
    - tensorrt==8.5.2
    - numpy==1.23.5
'''


r'''
TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work
TensorRT with DALI: https://github.com/NVIDIA/DL4AGX

#########################################################
How does TensorRT use memory?
1. The build phase
    It allocates device memory for timing layer implementations.
    Control the maximum amount of temporary memory the `memory pool limits` of the "builder config".
    The workspace size's default is the full size of device global memory.
    It requires to create buffers fir input, output, and weights.

2. The runtime phase
    runtimes quick start: https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/5.%20Understanding%20TensorRT%20Runtimes.ipynb

    An engine, on deserialization, allocates device memory to store the model weights.
    `ExecutionContext` uses two kinds of device memory:
    1. Persistent memory
    2. Scratch memory: for intermediate activation tensors / temporary storage. Controled by `set_memory_pool_limits`
>> check device memory in use: cudaGetMemInfo ???

[NVIDIA CUDA documentation]
CUDA lazy loading: it is enabled by setting the env variable `CUDA_MODULE_LOADING=LAZY`
    - reduce device memory usage
    - speed up TensorRT initialization


Threading: # TODO


'''

# set certain device
set_device_success = cudart.cudaSetDevice(1)
print('set cuda device:', set_device_success)

# model_path = '/home/doriskao/project/kaggle_bird/onnx_models/effb1_best_score_center_f1.onnx'
# input_shape = (1, 1, 256, 313)
# output_shape = (1, 182)

# yolov5 model
model_path = '/home/doriskao/workspace/test_edge/onnx_models/yolov5_post_nms_xyxy_v2_debug.onnx'
input_shape = (16128, 6)
# output_shape = (1, 16128, 6)
# model = onnx.load(model_path)


r'''
model = onnx.load(model_path)
engine = backend.prepare(model, device='CUDA:1')
input_data = np.random.random(size=(1, 3, 512, 512)).astype(np.float32)
output_data = engine.run(input_data)[0]

print(output_data)
print(output_data.shape)
'''


# parser
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# tensorrt 9.0.1.post12.dev4
# trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH must be set when use TensorRT version 8 or 9
network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_creation_flag)

# ONNX tensorrt backend
parser = trt.OnnxParser(network, logger)
# use the native `InstanceNormalization` implementation instead of the plugin one
# parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)  # tensorrt >= v8.6
# success = parser.parse(model.SerializeToString())
success = parser.parse_from_file(model_path)
print('parser:', success)
print(parser.get_error(0))


# set up optimization profile
profile = builder.create_optimization_profile()
profile.set_shape('input', input_shape, input_shape, input_shape)  # the name of the input tensor should be the same as the name of the input tensor in onnx model
# profile.set_shape('output0', input_shape, input_shape, input_shape)
# profile.set_shape('output1', (1,), (10,), (10,))
# set builder config: tensorrt.tensorrt.IBuilderConfig
builder_config = builder.create_builder_config()
pool_type = trt.MemoryPoolType.WORKSPACE
builder_config.set_memory_pool_limit(pool_type, int(2 * 1e9))  # 2GB
# create a version-compatible engine
# builder_config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)  # tensorrt >= version9.0
builder_config.flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
# add optimization profile
is_valid = builder_config.add_optimization_profile(profile)
print('optimization profile valid:', is_valid)

# build the tensorrt engine(tensorrt.ICudaEngine): (> tensorrt 9.0 use build_serialized_network)
# create an engine from a builder
# engine = builder.build_engine(network, builder_config)  # old version, deprecated after version 8
engine_bytes = builder.build_serialized_network(network, builder_config)  # return IHostMemory
print('network output:', network.num_outputs)

# load engine from file
# with open('trt_engine/example_trt852.engine', 'rb') as f:
    # engine_bytes = f.read()
# print('load engine bytes')
runtime = trt.Runtime(logger)
# Indicate to TensorRT that you trust the plan
# runtime.engine_host_code_allowed = True  # after version 9.0
# after creating the engine, builder, network, parser, andf build config can be destroied
engine = runtime.deserialize_cuda_engine(engine_bytes)  # return ICudaEngine
print('engine:', engine)

# save engine
# with open('trt_engine/example_trt852.engine', 'wb') as f:
    # f.write(engine_bytes)
# print('engine saved')

# create tensorrt execution context
r'''
context
similar to the process in CPU
CUDA stream: a linear sequence of execution that belongs to a specific device
    - torch.device: where to allocate the stream on
'''
context = engine.create_execution_context()  # return IExecutionContext

import tensorrt_utils as trt_utils
# allocate all buffers required for an engine. host/device inputs/outputs
inputs, outputs, bindings, stream = trt_utils.allocate_buffers(engine)
print('allocated buffers')

# input_data = np.random.randn(1, 1, 256, 313)
input_data = np.random.randn(1, 16128, 6)
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
print(len(trt_outputs))
pred = trt_outputs[0]
pred = pred.reshape(10, 6)
# nms_id = trt_outputs[1]
# pred = pred.reshape(1, 16128, 6)


@atexit.register
def free_buffer():
    print('free buffers')
    trt_utils.free_buffers(inputs, outputs, stream)
print('result:', pred.shape)


# prepare data
# tutorial: https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
# allocate memory for input and output buffer
# output = np.empty([1, 182], dtype=np.float32)
# allocate device memory
# d_input = cuda.mem_alloc(1 * input_data.nbytes)
# d_output = cuda.mem_alloc(1* output.nbytes)
# bindings = [int(d_input), int(d_output)]
# stream = cuda.Stream()

# for i in range(engine.num_io_tensors):
    # context.set_tensor_address(engine.get_tensor_name(i), bindings[i])


# input_buf = trt.cuda.alloc_buffer(
    # builder.max_batch_size * trt.volume(input_shape) * trt.float32.itemsize
# )
# output_buf = trt.cuda.alloc_buffer(
    # builder.max_batch_size * trt.volume(output_shape) * trt.float32.itemsize
# )

# run inference
# input_data = np.random.randn(1, 1, 256, 313)
# output_data = np.empty(output_shape, dtype=np.float32)
# input_buf.host = input_data.ravel()
# trt_outputs = [output_buf.device]
# trt_inputs = [input_buf.device]

# context.execute_async_v3(stream_handle=trt.cuda.Stream())
# output_buf.device_to_host()
# output_data[:] = np.reshape(output_buf.host, output_shape)
# print(output_data)
