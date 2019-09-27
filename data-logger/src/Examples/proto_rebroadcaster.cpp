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
#include "f1_datalogger/proto/TimestampedPacketMotionData.pb.h"
#include <boost/asio.hpp>

namespace scl = SL::Screen_Capture;
using boost::asio::ip::udp;




class ProtoRebroadcaster_2018DataGrabHandler : public deepf1::IF12018DataGrabHandler
{
public:
  ProtoRebroadcaster_2018DataGrabHandler(std::string host, unsigned int port)
  : host_(host), port_(port), socket(io_context)
  {
    receiver_endpoint = udp::endpoint(boost::asio::ip::address_v4::from_string(host_), port_);
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
    ready_ = false;
    deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData timestamped_packet_pb;
    timestamped_packet_pb.mutable_udp_packet()->CopyFrom(deepf1::twenty_eighteen::TwentyEighteenUDPStreamUtils::toProto(data.data));
    std::chrono::duration<double, std::milli> dt = data.timestamp - begin;
    timestamped_packet_pb.set_timestamp(dt.count());
    size_t num_bytes = timestamped_packet_pb.ByteSize();
    std::unique_ptr<char[]> buffer(new char[num_bytes]);
    timestamped_packet_pb.SerializeToArray(buffer.get(),num_bytes);
    boost::system::error_code error;
    size_t len = socket.send_to(boost::asio::buffer(buffer.get(),num_bytes), receiver_endpoint);
    //std::cout << "Sent motion packet of " << len << " bytes." << std::endl;
    ready_ = true;
    
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) override
  {
    
  }
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override
  {
    socket.open(udp::v4());
    ready_ = true;
    this->begin = begin;
  }
private:
  bool ready_;
  std::chrono::high_resolution_clock::time_point begin;
  std::string host_;
  unsigned int port_;
  boost::asio::io_context io_context;
  udp::socket socket;
  udp::endpoint receiver_endpoint;
};
class ProtoRebroadcaster_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
  ProtoRebroadcaster_FrameGrabHandler(std::string host, unsigned int port, std::vector<uint32_t> roi)
  : host_(host), port_(port), socket(io_context), roi_(roi)
  {
    receiver_endpoint = udp::endpoint(boost::asio::ip::address_v4::from_string(host_), port_);
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
      //std::printf("Extracting ROI: %u %u %u %u from image of size %u %u\n", x, y, w, h, data.image.rows, data.image.cols);
      cv::resize(data.image(cv::Range(y , y + h), cv::Range( x , x+w ) ),im_resize,cv::Size(200,66), cv::INTER_AREA);
      deepf1::protobuf::images::Image im_proto = deepf1::OpenCVUtils::cvimageToProto(im_resize);
      size_t num_bytes = im_proto.ByteSize();
      std::unique_ptr<char[]> buffer(new char[num_bytes]);
      im_proto.SerializeToArray(buffer.get(),num_bytes);
      boost::system::error_code error;
     // std::cout << "Sending image" << std::endl;
      size_t len = socket.send_to(boost::asio::buffer(buffer.get(),num_bytes), receiver_endpoint);
      //std::cout << "Sent image of " << len << " bytes." << std::endl;
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
    std::cout << "Opening image socket" << std::endl;
    socket.open(udp::v4());
    std::cout << "Opened image socket" << std::endl;
    ready = true;
    window_made=false;
  }
private:
  std::string host_;
  std::vector<uint32_t> roi_;
  unsigned int port_;
  bool ready, window_made;
  float resize_factor_;
  boost::asio::io_context io_context;
  udp::socket socket;
  udp::endpoint receiver_endpoint;
};

int main(int argc, char** argv)
{
  std::string search = "F1";
  float scale_factor=1.0;
  uint32_t x,y,w,h;
  x = std::stoi(std::string(argv[1]));
  y = std::stoi(std::string(argv[2]));
  w = std::stoi(std::string(argv[3]));
  h = std::stoi(std::string(argv[4]));
  
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

