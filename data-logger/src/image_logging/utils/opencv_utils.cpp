/*
 * opencv_utils.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/image_logging/utils/opencv_utils.h"
#include <iostream>
#include <cstring>
#include <string.h>
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
  memcpy((void *) rtn.data, (void *)(&(proto_image.image_data()[0])), rtn.step[0] * (size_t)rtn.rows );
  return rtn;
}
deepf1::protobuf::images::Image OpenCVUtils::cvimageToProto(const cv::Mat& cv_image)
{
  deepf1::protobuf::images::Image proto_image;
  //std::cerr << "Image Channels: " << channels << std::endl;
  deepf1::protobuf::images::ChannelOrder co = imTypeToProto(cv_image.type());
//  std::cerr << "Converted Channel Order" << std::endl;
  proto_image.set_channel_order(co);
  proto_image.set_cols(cv_image.cols);
  proto_image.set_rows(cv_image.rows);
  size_t totalsize = cv_image.step[0] * cv_image.rows;
  proto_image.mutable_image_data()->resize(totalsize);
  memcpy((void *) proto_image.mutable_image_data()->data(), (void *) cv_image.data, totalsize);
  return proto_image;
}
void OpenCVUtils::toCV(const scl::Image& image_scl, const scl::Point& size, cv::Mat& out)
{
  if(size.x>0 && size.y>0)
  {
    out.create( size.y,  size.x , CV_8UC4);
  }
  else
  {
    out.create( scl::Height(image_scl),  scl::Width(image_scl) , CV_8UC4);
  }
  scl::Extract(image_scl, out.data, out.step[0] * out.rows  * sizeof(scl::ImageBGRA));
}
cv::Mat OpenCVUtils::toCV(const scl::Image& image_scl, const scl::Point& size)
{
  cv::Mat out;
  OpenCVUtils::toCV(image_scl, size, out);
  return out;
}
} /* namespace deepf1 */
