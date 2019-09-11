/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */
#include "f1_datalogger/f1_datalogger.h"
//#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <Eigen/Geometry>
#include "f1_datalogger/udp_logging/utils/eigen_utils.h"
#include "f1_datalogger/image_logging/utils/opencv_utils.h"
#include <google/protobuf/util/json_util.h>
#include <fstream>
#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"

#include <grpcpp/grpcpp.h>

#include "f1_datalogger/proto/DeepF1_RPC.grpc.pb.h"
namespace scl = SL::Screen_Capture;
using boost::asio::ip::tcp;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using deepf1::protobuf::images::SendImageRequest;
using deepf1::protobuf::images::SendImageResponse;
using deepf1::protobuf::images::SendImage;
using deepf1::protobuf::images::SendMotionDataResponse;
using deepf1::protobuf::images::SendMotionDataRequest;
using deepf1::protobuf::images::SendMotionData;




class ProtoRebroadcaster_2018DataGrabHandler : public deepf1::IF12018DataGrabHandler
{
public:
  ProtoRebroadcaster_2018DataGrabHandler(std::string host, unsigned int port)
  : host_(host), port_(port)
  {
  }
  bool isReady() override
  {
    return ready_;
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data) override
  {
    deepf1::twenty_eighteen::protobuf::PacketMotionData datapb =  deepf1::twenty_eighteen::TwentyEighteenUDPStreamUtils::toProto(data.data);
    SendMotionDataRequest request;
    SendMotionDataResponse response;
    request.mutable_motion_data()->CopyFrom(datapb);
    ClientContext context;
    Status status = motion_stub_->SendMotionData(&context, request, &response);
    
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) override
  {
    
  }
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override
  {
    motion_channel = grpc::CreateChannel(
      host_+":"+std::to_string(port_), grpc::InsecureChannelCredentials());
    motion_stub_ = SendMotionData::NewStub(motion_channel);
    ready_ = true;
    this->begin = begin;
  }
private:
  bool ready_;
  std::chrono::high_resolution_clock::time_point begin;
  std::shared_ptr<Channel> motion_channel;
  std::unique_ptr<SendMotionData::Stub> motion_stub_;
  std::string host_;
  unsigned int port_;

};
class ProtoRebroadcaster_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
  ProtoRebroadcaster_FrameGrabHandler(std::string host, unsigned int port, std::vector<uint32_t> roi)
  : host_(host), port_(port), roi_(roi)
  {
  }
  virtual ~ProtoRebroadcaster_FrameGrabHandler()
  {
  }
  bool isReady() override
  {
    return ready;
  }
  void handleData(const deepf1::TimestampedImageData& data) override
  {
    try
    {
      ready = false;
      //0 32 1755 403
      uint32_t x = roi_[0];
      uint32_t y = roi_[1];
      uint32_t w = roi_[2];
      uint32_t h = roi_[3];
      cv::Mat im_resize;
      cv::resize(data.image(cv::Range(y , y + h), cv::Range( x , x+w ) ),im_resize,cv::Size(200,66), cv::INTER_AREA);
      deepf1::protobuf::images::Image im_proto = deepf1::OpenCVUtils::cvimageToProto(im_resize);
      SendImageRequest request;
      request.mutable_impb()->CopyFrom(im_proto);
      SendImageResponse response;
      //std::cout<<"Making request"<<std::endl;
      ClientContext context;
      Status status = stub_->SendImage(&context, request, &response);

      ready = true;
     // ready = false;
    }
    catch(std::exception& e)
    {
      std::cout<<std::string(e.what());
    }
  }
  void init(const deepf1::TimePoint& begin, const cv::Size& window_size) override
  {
    channel = grpc::CreateChannel(
      host_+":"+std::to_string(port_), grpc::InsecureChannelCredentials());
    stub_ = SendImage::NewStub(channel);
    ready = true;
    window_made=false;
  }
private:
  std::shared_ptr<Channel> channel;
  std::unique_ptr<SendImage::Stub> stub_;
  std::string host_;
  std::vector<uint32_t> roi_;
  unsigned int port_;
  bool ready, window_made;
  float resize_factor_;
};

int main(int argc, char** argv)
{
  std::string search = "F1";
  float scale_factor=1.0;
  uint32_t x,y,w,h;
  search = std::string(argv[1]);
  x = std::stoi(std::string(argv[2]));
  y = std::stoi(std::string(argv[3]));
  w = std::stoi(std::string(argv[4]));
  h = std::stoi(std::string(argv[5]));
  
  std::shared_ptr<ProtoRebroadcaster_FrameGrabHandler> image_handler(new ProtoRebroadcaster_FrameGrabHandler("127.0.0.1", 50051, std::vector<uint32_t>{x,y,w,h}));
  std::shared_ptr<ProtoRebroadcaster_2018DataGrabHandler> udp_handler(new ProtoRebroadcaster_2018DataGrabHandler("127.0.0.1", 50052));
  std::string inp;
  deepf1::F1DataLogger dl(search);  
  dl.start(60.0, udp_handler, image_handler);
  std::cout<<"Ctl-c to exit."<<std::endl;
  while (true)
  {
	  std::this_thread::sleep_for( std::chrono::seconds( 5 ) );
  }

}

