###
 # @Author: zhouyuchong
 # @Date: 2024-05-21 16:47:58
 # @Description: 
 # @LastEditors: zhouyuchong
 # @LastEditTime: 2024-05-21 16:54:29
### 

log() {
    local message="$1"
    current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${current_time}] $message"
}

CUR_DIR=$PWD

log "backing up"
mkdir backup
mv /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so ./backup/libnvds_infer.so
mv /opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_infer.so ./backup/libnvdsgst_infer.so
mv /opt/nvidia/deepstream/deepstream/sources/includes/nvdsinfer.h ./backup/nvdsinfer.h

export CUDA_VER=11.6

cp ./nvdsinfer.h /opt/nvidia/deepstream/deepstream/sources/includes/nvdsinfer.h

cd nvdsinfer
make -j$(nproc)
make install

cd $CUR_DIR
cd gst-nvinfer
make -j$(nproc)
make install

log "install success"



