#ifndef _DAMONZZZ_ALIGNER_H_
#define _DAMONZZZ_ALIGNER_H_

#include "opencv2/opencv.hpp"


namespace alignnamespace {
class Aligner {
public:
	Aligner();
	~Aligner();

	cv::Mat AlignFace(const cv::Mat & dst);
	cv::Mat AlignPlate(const cv::Mat & dst, int model_type);
	
private:
	class Impl;
	Impl* impl_;
};

} // namespace alignnamespace

#endif // !_DAMONZZZ_ALIGNER_H_