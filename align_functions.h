#ifndef _FACE_ALIGNER_H_
#define _FACE_ALIGNER_H_

#include "opencv2/opencv.hpp"


namespace mirror {
class Aligner {
public:
	Aligner();
	~Aligner();

	cv::Mat AlignFace(const cv::Mat & src, const cv::Mat & dst);
	cv::Mat AlignPlate(const cv::Mat & src, const cv::Mat & dst);
	
private:
	class Impl;
	Impl* impl_;
};

}

#endif // !_FACE_ALIGNER_H_