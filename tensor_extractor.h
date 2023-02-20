#ifndef _DAMONZZZ_EXTRACTOR_H_
#define _DAMONZZZ_EXTRACTOR_H_

#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "opencv2/opencv.hpp"

#define CLIP(a, min, max) (MAX(MIN(a, max), min))
#define CONF_THRESH 0.1
#define VIS_THRESH 0.75
#define NMS_THRESH 0.4

#define FACE_NETWIDTH 640
#define FACE_NETHEIGHT 480
#define PLATE_NETWIDTH 1920
#define PLATE_NETHEIGHT 1080

#define STREAMMUX_WIDTH 3392
#define STREAMMUX_HEIGHT 2000

static constexpr int LOCATIONS = 4;
static constexpr int FACE_ANCHORS = 10;
static constexpr int PLATE_ANCHORS = 8;

struct alignas(float) FaceInfo{
    float bbox[LOCATIONS];
    float confidence;
    float landmark[FACE_ANCHORS];
};

struct alignas(float) PlateInfo{
    float bbox[LOCATIONS];
    float confidence;
    float landmark[PLATE_ANCHORS];
};

struct anchorBox {
    float cx;
    float cy;
    float sx;
    float sy;
};

namespace extractornamespace {
class Extractor {
public:
	Extractor();
	~Extractor();

	void facelmks (NvDsMetaList * l_user, std::vector<FaceInfo>& res);
	bool platelmks(NvDsMetaList * l_user, std::vector<PlateInfo>& res);
	
private:
	class Impl;
	Impl* impl_;
};

}// namespace extractornamespace

#endif // !_DAMONZZZ_EXTRACTOR_H_