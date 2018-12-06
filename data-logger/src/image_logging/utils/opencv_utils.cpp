/*
 * opencv_utils.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "image_logging/utils/opencv_utils.h"

namespace deepf1
{

namespace scl = SL::Screen_Capture;
OpenCVUtils::OpenCVUtils()
{
}

OpenCVUtils::~OpenCVUtils()
{
}
cv::Mat OpenCVUtils::toCV(const scl::Image& image_scl)
{
  cv::Mat rtn;
  unsigned int height = scl::Height(image_scl);
  unsigned int width = scl::Width(image_scl);
  rtn.create(height, width, CV_8UC4);
  unsigned int pixel_size = sizeof(scl::ImageBGRA);
  scl::Extract(image_scl, rtn.data, height * width * pixel_size);
  return rtn;
}
} /* namespace deepf1 */
