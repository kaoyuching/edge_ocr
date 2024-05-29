# Edge

## Dependency
1. cuda-toolkit

```shell=
$ conda install nvidia/label/cuda-11.7.0::cuda-toolkit
```

## Dependency for Jetson Orin Nano (host: aarch64 / Jetpack 5.1.2):
- python 3.8.10
- cuda-toolkit

```shell=
$ conda install conda install nvidia/label/cuda-11.4.1::cuda-toolkit
```

- tensorrt == 8.5.2
- numpy == 1.23.5


## Onnx
Number plate detection with yolov5:
- `yolov5s.onnx`: Input image shape is (batch, 3, 512, 512). Output boxes shape is (batch, 16128, 6), where boxes result include (xmin, ymin, xmax, ymax, confidence, class_confidence)
- `yolov5_post_nms_xyxy_single.onnx`: Input shape is (1, 16128, 6). Output shape is (10, 6). The output is the top 10 boxes.
- `crnn.onnx`: Input image shape is (batch, 3, 64, 512). Output shape is (32, 1, 37).


## TensorRT
Setting environment variable `CUDA_MODULE_LOADING='LAZY'` can speed up TensorRT initialization and reduce device memory usage.
Run TensorRT example:

```shell=
$ CUDA_MODULE_LOADING='LAZY' python ./tensorrt/onnx_to_tensorrt.py
```


## Reference
1. NVIDIA TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
2. TensorRT python API documentation (v10.0.1): https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html
3. TensorRT python API documentation (v8.5.2): https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-852/api/python_api/index.html
4. TensorRT python API documentation (v7.3.1): https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/api/python_api/index.html
5. NVIDIA TensorRT release notes: https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html
