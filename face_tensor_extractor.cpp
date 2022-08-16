#include "face_tensor_extractor.h"


namespace extractornamespace {
class Extractor::Impl {
public:
	void facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res);
	// cv::Mat AlignPlate(const cv::Mat& dst, const cv::Mat& src);


private:
    
    // bool cmp(FaceInfo& a, FaceInfo& b);
    float iou(float lbox[4], float rbox[4]);
    void nms_and_adapt(std::vector<FaceInfo>& det, std::vector<FaceInfo>& res, float nms_thresh, int width, int height);
    void create_anchor_retinaface(std::vector<FaceInfo>& res, float *output, float conf_thresh, int width, int height);
    
};

Extractor::Extractor() {
    impl_ = new Impl();
}

Extractor::~Extractor() {
    if (impl_) {
        delete impl_;
    }
}

void Extractor::facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res) {
    return impl_->facelmks(l_user, res);
}

void Extractor::Impl::facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res) {
    static guint use_device_mem = 0;
    for (;l_user != NULL; l_user = l_user->next) { 
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
        continue; 
        }
        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
        NvDsInferLayerInfo *info = &meta->output_layers_info[0];
        info->buffer = meta->out_buf_ptrs_host[0];
        if (use_device_mem && meta->out_buf_ptrs_dev[0]) {
        // get all data from gpu to cpu
        cudaMemcpy (meta->out_buf_ptrs_host[0], meta->out_buf_ptrs_dev[0],
            info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
        std::vector < NvDsInferLayerInfo > outputLayersInfo (meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);
        float *output = (float*)(outputLayersInfo[0].buffer);
        std::vector<FaceInfo> temp;
        // std::vector<Detection> res;
        create_anchor_retinaface(temp, output, CONF_THRESH, NETWIDTH, NETHEIGHT);
        nms_and_adapt(temp, res, NMS_THRESH, NETWIDTH, NETHEIGHT);
    }  
}

bool cmp(FaceInfo& a, FaceInfo& b) {
    return a.score > b.score;
}

float Extractor::Impl::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

void Extractor::Impl::nms_and_adapt(std::vector<FaceInfo>& det, std::vector<FaceInfo>& res, float nms_thresh, int width, int height) {
    std::sort(det.begin(), det.end(), cmp);
    for (unsigned int m = 0; m < det.size(); ++m) {
        auto& item = det[m];
        res.push_back(item);
        for (unsigned int n = m + 1; n < det.size(); ++n) {
            if (iou(item.bbox, det[n].bbox) > nms_thresh) {
                det.erase(det.begin()+n);
                --n;
            }
        }
    }
    // crop larger area for better alignment performance 
    // there I choose to crop 50 more pixel 
    for (unsigned int m = 0; m < res.size(); ++m) {
        res[m].bbox[0] = CLIP(res[m].bbox[0]-25, 0, width - 1);
        res[m].bbox[1] = CLIP(res[m].bbox[1]-25 , 0, height -1);
        res[m].bbox[2] = CLIP(res[m].bbox[2]+25, 0, width - 1);
        res[m].bbox[3] = CLIP(res[m].bbox[3]+25, 0, height - 1);
    }

}

void Extractor::Impl::create_anchor_retinaface(std::vector<FaceInfo>& res, float *output, float conf_thresh, int width, int height) {
    int det_size = sizeof(FaceInfo) / sizeof(float);
    for (int i = 0; i < output[0]; i++){
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        
        FaceInfo det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        det.bbox[0] = CLIP(det.bbox[0], 0, width - 1);
        det.bbox[1] = CLIP(det.bbox[1] , 0, height -1);
        det.bbox[2] = CLIP(det.bbox[2], 0, width - 1);
        det.bbox[3] = CLIP(det.bbox[3], 0, height - 1);
        // det
        res.push_back(det);
        
    }
}


}
