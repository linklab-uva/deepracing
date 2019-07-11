/*
 * opencv_utils.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/image_logging/utils/opencv_utils.h"
#include <iostream>
#include <stdexcept>
namespace deepf1
{

namespace scl = SL::Screen_Capture;
OpenCVUtils::OpenCVUtils()
{
}

OpenCVUtils::~OpenCVUtils()
{
}
std::pair<deepf1::protobuf::images::ChannelOrder, uint32_t> OpenCVUtils::imTypeToProto(const int& cv_type)
{
  switch (cv_type)
  {
  case CV_8U:
  {
    return std::make_pair(deepf1::protobuf::images::ChannelOrder::GRAYSCALE, 1);
  }
  case CV_8UC4:
  {
    return std::make_pair(deepf1::protobuf::images::ChannelOrder::BGRA, 1);
  }
  case CV_8UC3:
  {
    return std::make_pair(deepf1::protobuf::images::ChannelOrder::BGR, 1);
  }
  case CV_16U:
  {
    return std::make_pair(deepf1::protobuf::images::ChannelOrder::GRAYSCALE, 2);
  }
  case CV_16UC4:
  {
    return std::make_pair(deepf1::protobuf::images::ChannelOrder::BGRA, 2);
  }
  case CV_16UC3:
  {
    return std::make_pair(deepf1::protobuf::images::ChannelOrder::BGR, 2);
  }
  default:
    throw std::runtime_error("Unsupported image type: " + std::to_string(cv_type));
  }
}
deepf1::protobuf::images::Image OpenCVUtils::imageToProto(const cv::Mat& cv_image)
{
  deepf1::protobuf::images::Image proto_image;
  std::pair<deepf1::protobuf::images::ChannelOrder, uint32_t> metadata = imTypeToProto(cv_image.type());
  proto_image.set_channel_order(metadata.first);
  proto_image.set_bytewidth(metadata.second);
  proto_image.set_cols(cv_image.cols);
  proto_image.set_rows(cv_image.rows);
  //proto_image.mutable_image_data()->copy((char*)cv_image.data, cv_image.total() * cv_image.elemSize());
  return proto_image;
}
void OpenCVUtils::toCV(const scl::Image& image_scl, const scl::Point& size, cv::Mat& out)
{
	out.create(size.y, size.x, CV_8UC4);
	scl::Extract(image_scl, out.data, out.rows * out.cols * sizeof(scl::ImageBGRA));
}
cv::Mat OpenCVUtils::toCV(const scl::Image& image_scl, const scl::Point& size)
{
  cv::Mat rtn;
  unsigned int rows = scl::Height(image_scl), cols = scl::Width(image_scl);
  //std::cout << "Got an image of Height: " << rows << " and Width: " << cols << std::endl;
  rtn.create(rows, cols, CV_8UC4);
  scl::Extract(image_scl, rtn.data, rtn.rows * rtn.cols * sizeof(scl::ImageBGRA));
  return rtn;
}
} /* namespace deepf1 */
