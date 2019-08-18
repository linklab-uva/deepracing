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
namespace scl = SL::Screen_Capture;class OpenCV_Viewer_Example_2018DataGrabHandler : public deepf1::IF12018DataGrabHandler
{
public:
  OpenCV_Viewer_Example_2018DataGrabHandler()
  {
    car_index = 0;
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
	//  std::printf("Got a telemetry packet. Steering Ratio: %d\n", data.data.m_carTelemetryData[car_index].m_steer);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data) override
  {
	//  std::printf("Got an event packet %s.\n", (data.data.m_eventStringCode));
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data) override
  {
    //ready_ = false;
    //const deepf1::twenty_eighteen::CarMotionData& motionPacket = data.data.m_carMotionData[car_index];
    //Eigen::Affine3d poseGlobal = deepf1::EigenUtils::motionPacketToPose(motionPacket);
    //Eigen::Vector3d velocityGlobal(motionPacket.m_worldVelocityX, motionPacket.m_worldVelocityY, motionPacket.m_worldVelocityZ);
    //Eigen::Vector3d velocityLocalComputed = poseGlobal.rotation().inverse() * velocityGlobal;
    //Eigen::Vector3d velocityLocal(data.data.m_localVelocityX, data.data.m_localVelocityY, data.data.m_localVelocityZ);
    //std::cout << std::endl;
    //std::cout << "Velocity Local Computed: " << std::endl << velocityLocalComputed << std::endl;
    //std::cout << "Velocity Local: " << std::endl << velocityLocal << std::endl;
    //std::cout << "Velocity Diff: " << (velocityLocalComputed - velocityLocal).norm() << std::endl;
    //std::cout  << std::endl;
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //ready_ = true;

    

	  
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) override
  {
	  if (data.data.m_isSpectating)
	  {
		car_index = data.data.m_spectatorCarIndex;
	  }
	  else
	  {
		  car_index = 0;
	  }
    
  }
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override
  {
    ready_ = true;
	  car_index = 0;
    this->begin = begin;
  }
private:
  bool ready_;
  std::chrono::high_resolution_clock::time_point begin;
  uint8_t car_index;
  float t1 = 0.0;
  float t2 = 0.0;
};
class OpenCV_Viewer_Example_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
  OpenCV_Viewer_Example_FrameGrabHandler() :
    window_name("captured_window")
  {
	  
  }
  virtual ~OpenCV_Viewer_Example_FrameGrabHandler()
  {
//	video_writer_->release();
  //  cv::destroyWindow(window_name);
	  running = false;
  }
  bool isReady() override
  {
    return ready;
  }
  void handleData(const deepf1::TimestampedImageData& data) override
  {
    if (!window_made)
    {
      cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
      window_made = true;
    }
    ready = false;
    cv::imshow(window_name, data.image);
    cv::waitKey(5);
    ready = true;
//	std::chrono::duration<double> d = data.timestamp - begin;
//	std::cout << "Got an image with timestamp "<< d.count() << std::endl;
  }
  void pulseReady()
  {
	  unsigned int sleeptime = 500;
	  while (running)
	  {
		  if (!ready)
		  {
			  ready = true;
			  std::this_thread::sleep_for(std::chrono::milliseconds(sleeptime));  
		  }
	  }
  }
  void init(const deepf1::TimePoint& begin, const cv::Size& window_size) override
  {
    imcount = 0;
	  running = true;
	  ready = true;
    window_made = false;
	  //readyThread = std::thread(std::bind(&OpenCV_Viewer_Example_FrameGrabHandler::pulseReady, this));
	  this->begin = deepf1::TimePoint(begin);
	  before = deepf1::TimePoint(begin);
	  after = deepf1::TimePoint(begin);
	  video_writer_.reset(new cv::VideoWriter("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), captureFreq, window_size));
  }
  static constexpr float captureFreq = 30.0;
  const std::string window_name;
private:
  std::shared_ptr<cv::VideoWriter> video_writer_;
  deepf1::TimePoint begin;
  deepf1::TimePoint before, after;
  std::thread readyThread;
  bool ready;
  bool running;
  bool window_made;
  unsigned int imcount;
};

int main(int argc, char** argv)
{
  std::string search = "F1";
  if (argc > 1)
  {
    search = std::string(argv[1]);
  }
  std::shared_ptr<OpenCV_Viewer_Example_FrameGrabHandler> image_handler(new OpenCV_Viewer_Example_FrameGrabHandler());
  std::shared_ptr<OpenCV_Viewer_Example_2018DataGrabHandler> udp_handler(new OpenCV_Viewer_Example_2018DataGrabHandler());
  std::string inp;
  deepf1::F1DataLogger dl(search);  
  dl.start((double)OpenCV_Viewer_Example_FrameGrabHandler::captureFreq, udp_handler, image_handler);
  std::cout<<"Ctl-c to exit."<<std::endl;
  while (true)
  {
	  std::this_thread::sleep_for( std::chrono::seconds( 5 ) );
  }

}

