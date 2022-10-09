# Custom gst-nvinfer (DEMO)
This is a custom gst-nvinfer plugin to do some preprocess.

## Requirements
+ Deepstream 6.0+
+ Opencv

## Usage
1. set cuda environment

```
export CUDA_VER=11.6
```
2. compile Makefile

```
make
make install
```
NOTE: To compile the sources, run make with "sudo" or root permission.

3. set config file

first to set primary gie's output-tensor-meta to true. For example, in retinaface config file:
```
output-tensor-meta=1
```

use kyeword alignment and user-meta in next gie-config file. For example, in arcface config file:
```
alignment=1
user-meta=1
```
Now only retinaface and arcface(operate on retinaface)

## Details
the custom gst-nvinfer has two new properties: alignment and user-meta.

Once these properties are true, there would be a preprocess on every croped object. 

It first receives the output tensor meta, to extract the 5 landmarks of a face and do NMS. By `similarTransform`, a matrix `M` will be generated.

then use the croped object and M to do alignment: `cv::warpPerspective`.

finally, cover the original surface with the output of alignment.