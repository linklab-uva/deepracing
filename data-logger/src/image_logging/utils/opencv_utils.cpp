/*
 * opencv_utils.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/image_logging/utils/opencv_utils.h"

namespace deepf1
{

namespace scl = SL::Screen_Capture;
OpenCVUtils::OpenCVUtils()
{
}

OpenCVUtils::~OpenCVUtils()
{
}
cv::Mat OpenCVUtils::toCV(const scl::Image& image_scl, const scl::Point& size)
{
  cv::Mat rtn;
  if( size.y == 0 || size.x==0 )
  {
    rtn.create(scl::Height(image_scl), scl::Width(image_scl), CV_8UC4);
  }else
  {
    rtn.create(size.y, size.x, CV_8UC4);
  }
  scl::Extract(image_scl, rtn.data, rtn.rows * rtn.cols * sizeof(scl::ImageBGRA));
  return rtn;
}
} /* namespace deepf1 */
