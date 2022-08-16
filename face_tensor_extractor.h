#ifndef _DAMONZZZ_EXTRACTOR_H_
#define _DAMONZZZ_EXTRACTOR_H_

#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "opencv2/opencv.hpp"

#define CLIP(a, min, max) (MAX(MIN(a, max), min))
#define CONF_THRESH 0.1
#define VIS_THRESH 0.9
#define NMS_THRESH 0.4
#define NETWIDTH 640
#define NETHEIGHT 640
static constexpr int LOCATIONS = 4;
static constexpr int ANCHORS = 10;

struct alignas(float) FaceInfo{
    float bbox[LOCATIONS];
    float score;
    float anchor[ANCHORS];
};

namespace extractornamespace {
class Extractor {
public:
	Extractor();
	~Extractor();

	void facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res);
	// cv::Mat AlignPlate(const cv::Mat & src, const cv::Mat & dst);
	
private:
	class Impl;
	Impl* impl_;
};

}// namespace extractornamespace

#endif // !_DAMONZZZ_EXTRACTOR_H_