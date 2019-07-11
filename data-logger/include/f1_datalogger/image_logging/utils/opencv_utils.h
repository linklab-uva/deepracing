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
#include "f1_datalogger/proto/Image.pb.h"
namespace deepf1
{
namespace scl = SL::Screen_Capture;
class OpenCVUtils
{
public:
  OpenCVUtils();
  virtual ~OpenCVUtils();
  static void toCV(const scl::Image& image_scl, const scl::Point& size, cv::Mat& out);
  static cv::Mat toCV(const scl::Image& image_scl, const scl::Point& size );
  static deepf1::protobuf::images::Image imageToProto(const cv::Mat& cv_image);
  static std::pair<deepf1::protobuf::images::ChannelOrder, uint32_t> imTypeToProto(const int& cv_type);
};

} /* namespace deepf1 */

#endif /* INCLUDE_OPENCV_UTILS_H_ */
