#include "tensor_extractor.h"


namespace extractornamespace {
class Extractor::Impl {
public:
	void facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res);
    bool platelmks(NvDsMetaList * l_user, std::vector<PlateInfo>& res);
	// cv::Mat AlignPlate(const cv::Mat& dst, const cv::Mat& src);


private:
    
    // bool cmp(FaceInfo& a, FaceInfo& b);
    // bool cmp(PlateInfo& a, PlateInfo& b);
    float iou(float lbox[4], float rbox[4]);
    void nms_and_adapt(std::vector<FaceInfo>& det, std::vector<FaceInfo>& res, float nms_thresh, int width, int height);
    bool nms_and_adapt_plate(std::vector<PlateInfo>& det, std::vector<PlateInfo>& res, float nms_thresh, int width, int height);

    void decode_bbox_retina_face(std::vector<FaceInfo>& res, float *output, float conf_thresh, int width, int height);
    void decode_bbox_retina_plate(std::vector<anchorBox> &anchor, std::vector<PlateInfo>& res, float *bbox, float *lmk, float *conf, 
                                float bbox_threshold, int width, int height);
    void create_anchor_retina_plate(std::vector<anchorBox> &anchor, int w, int h);
    
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

bool Extractor::platelmks(NvDsMetaList * l_user, std::vector<PlateInfo>& res) {
    return impl_->platelmks(l_user, res);
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
        decode_bbox_retina_face(temp, output, CONF_THRESH, FACE_NETWIDTH, FACE_NETHEIGHT);
        nms_and_adapt(temp, res, NMS_THRESH, FACE_NETWIDTH, FACE_NETHEIGHT);
    }  
    return;
}

bool Extractor::Impl::platelmks(NvDsMetaList * l_user, std::vector<PlateInfo>& res) {
    static guint use_device_mem = 1;
    bool flag = false;
    for (;l_user != NULL; l_user = l_user->next) { 
        // std::cout<<"222"<<std::endl;
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
            continue; 
        }
        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;

        // get bboxs
        NvDsInferLayerInfo *info_0 = &meta->output_layers_info[0];
        info_0->buffer = meta->out_buf_ptrs_host[0];
        if (use_device_mem && meta->out_buf_ptrs_dev[0]) {
            // get all data from gpu to cpu
            cudaMemcpy (meta->out_buf_ptrs_host[0], meta->out_buf_ptrs_dev[0],
                info_0->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
        // double* ptr = (double*)info->buffer;
        // for( size_t i=0; i<info->inferDims.numElements; i++ )
        // {
        //     std::cout << "Tensor " << i << ": " << ptr[i] << std::endl;
        // }
        // std::cout<<"copy: "<<info->inferDims.numElements * 4<<std::endl;
        // std::vector<NvDsInferLayerInfo> outputLayersInfo (meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);
        float *bbox = (float*)(info_0->buffer);

        //get lmks
        NvDsInferLayerInfo *info_1 = &meta->output_layers_info[1];
        info_1->buffer = meta->out_buf_ptrs_host[1];
        if (use_device_mem && meta->out_buf_ptrs_dev[1]) {
            // get all data from gpu to cpu
            cudaMemcpy (meta->out_buf_ptrs_host[1], meta->out_buf_ptrs_dev[1],
                info_1->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
                
        float *lmks = (float*)(info_1->buffer);

        //get lmks
        NvDsInferLayerInfo *info_2 = &meta->output_layers_info[2];
        info_2->buffer = meta->out_buf_ptrs_host[2];
        if (use_device_mem && meta->out_buf_ptrs_dev[2]) {
            // get all data from gpu to cpu
            cudaMemcpy (meta->out_buf_ptrs_host[2], meta->out_buf_ptrs_dev[2],
                info_2->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }    
        float *conf = (float*)(info_2->buffer);

        std::vector<anchorBox> anchor;
        std::vector<PlateInfo> temp;
        create_anchor_retina_plate(anchor, PLATE_NETWIDTH, PLATE_NETHEIGHT);
        decode_bbox_retina_plate(anchor, temp, bbox, lmks, conf, CONF_THRESH, PLATE_NETWIDTH, PLATE_NETHEIGHT);
        flag = nms_and_adapt_plate(temp, res, NMS_THRESH, PLATE_NETWIDTH, PLATE_NETHEIGHT);
    }  
    return flag;
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
    std::sort(det.begin(), det.end(), [](FaceInfo& a, FaceInfo& b){return a.confidence > b.confidence;});
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
    // there I choose to crop 20 more pixel 
    for (unsigned int m = 0; m < res.size(); ++m) {
        res[m].bbox[0] = CLIP(res[m].bbox[0], 0, width - 1);
        res[m].bbox[1] = CLIP(res[m].bbox[1], 0, height -1);
        res[m].bbox[2] = CLIP(res[m].bbox[2], 0, width - 1);
        res[m].bbox[3] = CLIP(res[m].bbox[3], 0, height - 1);
    }

}

bool Extractor::Impl::nms_and_adapt_plate(std::vector<PlateInfo>& det, std::vector<PlateInfo>& res, float nms_thresh, int width, int height) {
    std::sort(det.begin(), det.end(), [](PlateInfo& a, PlateInfo& b){return a.confidence > b.confidence;});
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
    // std::cout<<"after nms, size: "<<res.size()<<std::endl;
    // top k
    std::sort(res.begin(), res.end(), [](PlateInfo& a, PlateInfo& b){return a.confidence > b.confidence;});
    if(res.size() > 1){
        res.erase(res.begin()+1, res.end()); 

    }
    // std::cout<<"after topk, size: "<<res.size()<<std::endl;
    // if nothing extracted, return false
    if(res.size() != 0){
        return true;
    }
    else{
        return false;
    }
}

void Extractor::Impl::decode_bbox_retina_face(std::vector<FaceInfo>& res, float *output, float conf_thresh, int width, int height) {
    int det_size = sizeof(FaceInfo) / sizeof(float);
    for (int i = 0; i < output[0]; i++){
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        FaceInfo det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        det.bbox[0] = CLIP(det.bbox[0], 0, width - 1);
        det.bbox[1] = CLIP(det.bbox[1] , 0, height -1);
        det.bbox[2] = CLIP(det.bbox[2], 0, width - 1);
        det.bbox[3] = CLIP(det.bbox[3], 0, height - 1);
        res.push_back(det);
        
    }
}

void Extractor::Impl::create_anchor_retina_plate(std::vector<anchorBox> &anchor, int w, int h) {
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (unsigned int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {16, 32};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {64, 128};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {256, 512};
    min_sizes[2] = minsize3;

    for (unsigned int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (unsigned int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0 / w;
                    float s_ky = min_size[l] * 1.0 / h;
                    float cx = (j + 0.5) * steps[k] / w;
                    float cy = (i + 0.5) * steps[k] / h;
                    anchorBox axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }
    }
}

void Extractor::Impl::decode_bbox_retina_plate(std::vector<anchorBox> &anchor, std::vector<PlateInfo>& res, float *bbox, float *lmk, float *conf, 
                                float bbox_threshold, int width, int height) {
    for (unsigned int i = 0; i < anchor.size(); ++i) {
        // std::cout<<*(conf + 1)<<std::endl;
        if (*(conf + 1) > bbox_threshold) {
            anchorBox tmp = anchor[i];
            anchorBox tmp1;
        
            // decode bbox
            // std::cout<<tmp.cx<<" "<<tmp.cy<<" "<<tmp.sx<<" "<<tmp.sy<<std::endl;
            tmp1.cx = tmp.cx + *bbox * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(bbox + 1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(bbox + 2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(bbox + 3) * 0.2);

            PlateInfo det;
            det.bbox[0] = (tmp1.cx - tmp1.sx / 2) * width;
            det.bbox[1] = (tmp1.cy - tmp1.sy / 2) * height;
            det.bbox[2] = (tmp1.cx + tmp1.sx / 2) * width - det.bbox[0];
            det.bbox[3] = (tmp1.cy + tmp1.sy / 2) * height - det.bbox[1];

            det.bbox[0] = CLIP(det.bbox[0], 0, width - 1);
            det.bbox[1] = CLIP(det.bbox[1], 0, height -1); 
            det.bbox[2] = CLIP(det.bbox[2], 0, width - 1);
            det.bbox[3] = CLIP(det.bbox[3], 0, height - 1);
  
            det.confidence = *(conf + 1);
            
            for(unsigned int j = 0; j < 8; ){
                
                det.landmark[j]   = (tmp.cx + *(lmk + j) * 0.1 * tmp.sx) * width;
                det.landmark[j+1] = (tmp.cy + *(lmk + j + 1) * 0.1 * tmp.sy) * height;
                j = j + 2;
            }
            res.push_back(det);
        }
        
        bbox += 4;
        conf += 2;
        lmk  += 8;
        
    }
}

}
