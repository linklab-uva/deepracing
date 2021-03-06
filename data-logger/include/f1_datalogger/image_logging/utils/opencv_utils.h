/*
 * opencv_utils.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_OPENCV_UTILS_H_
#define INCLUDE_OPENCV_UTILS_H_
#include <f1_datalogger/image_logging/visibility_control.h>
#include <f1_datalogger/proto_dll_macro.h>
#include "ScreenCapture.h"
#include "opencv2/core.hpp"
#include "f1_datalogger/proto/Image.pb.h"

namespace deepf1
{
namespace scl = SL::Screen_Capture;
class F1_DATALOGGER_IMAGE_LOGGING_PUBLIC OpenCVUtils
{
public:
  typedef cv::Vec<uint8_t, 4> CVPixel;
  OpenCVUtils();
  virtual ~OpenCVUtils();

  static void toCV(const scl::Image& image_scl, const scl::Point& size, cv::Mat& out);
  static cv::Mat toCV(const scl::Image& image_scl, const scl::Point& size = scl::Point({0,0}));

  static deepf1::protobuf::images::Image cvimageToProto(const cv::Mat& cv_image);
  static cv::Mat protoImageToCV(const deepf1::protobuf::images::Image& proto_image);


  static deepf1::protobuf::images::ChannelOrder imTypeToProto(const int& cv_type);
  static int protoTypeToCV(const deepf1::protobuf::images::ChannelOrder& proto_type);

};

} /* namespace deepf1 */

#endif /* INCLUDE_OPENCV_UTILS_H_ */
