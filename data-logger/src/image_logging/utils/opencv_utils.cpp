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
  unsigned int height, width;
  if( size.y == 0 || size.x==0 )
  {
    height = scl::Height(image_scl);
    width = scl::Width(image_scl);
  }else
  {
    height = size.y;
    width = size.x;
  }
  rtn.create(height, width, CV_8UC4);
  scl::Extract(image_scl, rtn.data, height * width * sizeof(scl::ImageBGRA));
  return rtn;
}
} /* namespace deepf1 */
