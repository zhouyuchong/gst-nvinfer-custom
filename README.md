<!--
 * @Author: zhouyuchong
 * @Date: 2024-02-26 14:51:58
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-09-19 17:37:19
-->
# Custom gst-nvinfer (DEMO)
This is a custom gst-nvinfer plugin to do some preprocess and postprocess.

## Requirements
+ Deepstream 6.0+
+ Opencv

## Feature
+ add landmarks and number of lmks in `object_user_metadata`
+ use npp to do alignment
+ support multi batch size now

## How it works
### Detector
1. add landmarks in `nvdsinfer` so it can be processed together with bondingboxes in `nvdsinfer_customparser`. 
2. if `numLmks` not 0, we concat landmarks to the tail of *labels* (notice only modify function `DetectPostprocessor::fillUnclusteredOutput` in [`nvdsinfer_context_impl_output_parsing.cpp`](https://github.com/zhouyuchong/gst-nvinfer-custom/blob/40102b2ec323cf613ce202f213b31caff8189a52/nvdsinfer/nvdsinfer_context_impl_output_parsing.cpp#L490))
3. decode landmarks in `attach_metadata_detector` and attach them to `object_user_metadata`

### Classifier
1. decode landmarks in object_user_metadata
2. use npp to do alignment
3. Done!

## Usage
### 1. install
```
sh install.sh
```
this script will auto backup original `nvinfer` related lib to `./backup`.

### 2. modify `nvdsinfer_parser`
   set `oinfo.numLmks` and `oinfo.landmarks` properties in your `parse-bbox-func-name`.


### 3. set config file

#### Detector
set `cluster-mode=4` since we only modify `fillUnclusteredOutput`.
```
cluster-mode=4
enable-output-landmark = 1
```

#### Classifier
```
alignment-type=1
alignment-pics-path=/path/to/save/your/alignment_pics
```

alignment-type=1->[arcface](https://github.com/deepinsight/insightface)
alignment-type=2(not impl yet)->[Retina_License_Plate](https://github.com/gm19900510/Pytorch_Retina_License_Plate)

### 4. restore
```
sh restore.sh
```

## TODO
+ ~~cal affine matrix use cuSolver(not efficient as expected)~~
+ cal affine matrix use eigen
+ add more test case

## Reference
+ [retinaface & arcface](https://github.com/deepinsight/insightface)
+ [yolov8-face](https://github.com/derronqi/yolov8-face)
+ [NPP](https://docs.nvidia.com/cuda/archive/10.1/npp/index.html)