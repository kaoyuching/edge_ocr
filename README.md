# Edge

## TensorRT
Setting environment variable `CUDA_MODULE_LOADING='LAZY'` can speed up TensorRT initialization and reduce device memory usage.
Run TensorRT example:

```shell=
$ CUDA_MODULE_LOADING='LAZY' python ./tensorrt/onnx_to_tensorrt.py
```
