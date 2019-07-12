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
deepf1::protobuf::images::ChannelOrder OpenCVUtils::imTypeToProto(const int& cv_type)
{
  switch (cv_type)
  {
  case CV_8U:
  {
    return deepf1::protobuf::images::ChannelOrder::GRAYSCALE;
  }
  case CV_8UC4:
  {
    return deepf1::protobuf::images::ChannelOrder::BGRA;
  }
  case CV_8UC3:
  {
    return deepf1::protobuf::images::ChannelOrder::BGR;
  }
  default:
  {
    std::string err = "Unsupported image type: " + std::to_string(cv_type);
    std::cerr << err << std::endl;
    throw std::runtime_error(err);
  }
  }
}
int OpenCVUtils::protoTypeToCV(const deepf1::protobuf::images::ChannelOrder& proto_type)
{
  switch(proto_type)
  {
  case deepf1::protobuf::images::ChannelOrder::BGR:
  {
    return CV_8UC3;
  }
  case deepf1::protobuf::images::ChannelOrder::BGRA:
  {
    return CV_8UC4;
  }
  case deepf1::protobuf::images::ChannelOrder::GRAYSCALE:
  {
    return CV_8U;
  }
  default:
  {
    std::string err = "Unsupported image type: " + google::protobuf::GetEnumDescriptor< deepf1::protobuf::images::ChannelOrder >()->value(proto_type)->name();
    std::cerr << err << std::endl;
    throw std::runtime_error(err);
  }

  }
}
cv::Mat OpenCVUtils::protoImageToCV(const deepf1::protobuf::images::Image& proto_image)
{
  cv::Mat rtn;
  rtn.create(proto_image.rows(), proto_image.cols(), protoTypeToCV(proto_image.channel_order()));
  memcpy(rtn.data, (uchar *)(&(proto_image.image_data()[0])), (size_t)rtn.cols * (size_t)rtn.rows * (size_t)rtn.channels());
  return rtn;
}
deepf1::protobuf::images::Image OpenCVUtils::cvimageToProto(const cv::Mat& cv_image)
{
  deepf1::protobuf::images::Image proto_image;
  std::cerr << "Converting Channel Order: " << cv_image.type() << std::endl;
  uint32_t channels = cv_image.channels();
  std::cerr << "Image Channels: " << channels << std::endl;
  deepf1::protobuf::images::ChannelOrder co = imTypeToProto(cv_image.type());
  std::cerr << "Converted Channel Order" << std::endl;
  proto_image.set_channel_order(co);
  proto_image.set_cols(cv_image.cols);
  proto_image.set_rows(cv_image.rows);
  uint32_t totalsize = cv_image.step[0] * cv_image.rows;
  proto_image.mutable_image_data()->resize(totalsize);
  memcpy(proto_image.mutable_image_data()->data(), cv_image.data, totalsize);
  return proto_image;
}
void OpenCVUtils::toCV(const scl::Image& image_scl, const scl::Point& size, cv::Mat& out)
{
	out.create(size.y, size.x, CV_8UC4);
	scl::Extract(image_scl, out.data, (size_t)out.rows * (size_t)out.cols * sizeof(scl::ImageBGRA));
}
cv::Mat OpenCVUtils::toCV(const scl::Image& image_scl, const scl::Point& size)
{
  cv::Mat rtn;
  uint32_t rows = scl::Height(image_scl), cols = scl::Width(image_scl);
  //std::cout << "Got an image of Height: " << rows << " and Width: " << cols << std::endl;
  rtn.create(rows, cols, CV_8UC4);
  scl::Extract(image_scl, rtn.data, (size_t)cols * (size_t)rows * sizeof(scl::ImageBGRA));
  return rtn;
}
} /* namespace deepf1 */
