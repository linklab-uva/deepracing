/*
 * opencv_utils.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_OPENCV_UTILS_H_
#define INCLUDE_OPENCV_UTILS_H_
#include "ScreenCapture.h"
#include "opencv2/core.hpp"
namespace deepf1
{
namespace scl = SL::Screen_Capture;
class OpenCVUtils
{
public:
  OpenCVUtils();
  virtual ~OpenCVUtils();
  static cv::Mat toCV(const scl::Image& image_scl, const scl::Point& size = scl::Point() );
};

} /* namespace deepf1 */

#endif /* INCLUDE_OPENCV_UTILS_H_ */
