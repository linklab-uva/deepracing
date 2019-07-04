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
    return true;
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
	  //Eigen::Vector3d velocityGlobal(data.data.m_carMotionData[car_index].m_worldVelocityX, data.data.m_carMotionData[car_index].m_worldVelocityY, data.data.m_carMotionData[car_index].m_worldVelocityZ);
	  //Eigen::Vector3d velocityLocal(data.data.m_localVelocityX, data.data.m_localVelocityY, data.data.m_localVelocityZ);
	 

	  //Eigen::Vector3d forward(data.data.m_carMotionData[car_index].m_worldForwardDirX, data.data.m_carMotionData[car_index].m_worldForwardDirY, data.data.m_carMotionData[car_index].m_worldForwardDirZ);
	  //forward.normalize();
	  //Eigen::Vector3d right(data.data.m_carMotionData[car_index].m_worldRightDirX, data.data.m_carMotionData[car_index].m_worldRightDirY, data.data.m_carMotionData[car_index].m_worldRightDirZ);
	  //right.normalize();
	  //Eigen::Vector3d up = right.cross(forward);
	  //up.normalize();
	  //Eigen::Matrix3d rotMatrix = Eigen::Matrix3d::Identity();
	  //rotMatrix.col(0) = -right;
	  //rotMatrix.col(1) = up;
	  //rotMatrix.col(2) = forward;

	  //Eigen::Vector3d translation(data.data.m_carMotionData[car_index].m_worldPositionX, data.data.m_carMotionData[car_index].m_worldPositionY, data.data.m_carMotionData[car_index].m_worldPositionZ);
	  //Eigen::Affine3d pose = deepf1::EigenUtils::motionPacketToPose(data.data.m_carMotionData[car_index]);
	  //Eigen::Quaterniond rotation(pose.rotation());
	  //rotation.normalize();
	  //Eigen::Vector3d velocityLocalComputed = rotation.conjugate() * velocityGlobal;
	  //Eigen::Vector3d velocityGlobalComputed = rotation * velocityLocal;

	  //std::cout << std::endl;
	  //std::cout << "Global Velocity Given: " << std::endl << velocityGlobal << std::endl;
	  //std::cout << "Global Velocity Computed: " << std::endl << velocityGlobalComputed << std::endl;
	  //std::cout << "Global Velocity Diff: " << std::endl << (velocityGlobal - velocityGlobalComputed).norm() << std::endl;	
	  //std::cout << std::endl;


	  //std::cout << std::endl;
	  //std::cout << "Local Velocity Given: " << std::endl << velocityLocal << std::endl;
	  //std::cout << "Local Velocity Computed: " << std::endl << velocityLocalComputed << std::endl;
	  //std::cout << "Local Velocity Diff: " << std::endl << (velocityLocal - velocityLocalComputed).norm() << std::endl;
	  //std::cout << std::endl;

	  
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
	  if(bool(data.data.m_gamePaused))
	  {
		  std::cout << "Game is paused" << std::endl;
	  }
	  else
	  {
		  std::cout << "Game is not paused" << std::endl;
	  }
    
  }
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override
  {
	car_index = 0;
    this->begin = begin;
  }
private:
  std::chrono::high_resolution_clock::time_point begin;
  uint8_t car_index;
  float t1 = 0.0;
  float t2 = 0.0;
};
class OpenCV_Viewer_Example_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
  OpenCV_Viewer_Example_FrameGrabHandler()
   : window_name("cv_example")
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
	ready = false;
 //   cv::Mat img_cv_video;
 //   cv::cvtColor(data.image, img_cv_video, cv::COLOR_BGRA2BGR);
	//video_writer_->write(img_cv_video);
	cv::imshow(window_name, data.image);
	cv::waitKey(50);
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
	//readyThread = std::thread(std::bind(&OpenCV_Viewer_Example_FrameGrabHandler::pulseReady, this));
	this->begin = deepf1::TimePoint(begin);
	before = deepf1::TimePoint(begin);
	after = deepf1::TimePoint(begin);
	video_writer_.reset(new cv::VideoWriter("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), captureFreq, window_size));
  }
  static constexpr float captureFreq = 30.0;
private:
  std::shared_ptr<cv::VideoWriter> video_writer_;
  deepf1::TimePoint begin;
  std::string window_name;
  deepf1::TimePoint before, after;
  std::thread readyThread;
  bool ready;
  bool running;
  unsigned int imcount;
};

int main(int argc, char** argv)
{
  std::string search = "CMake";
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
	  std::this_thread::sleep_for(std::chrono::seconds(1));
  }

}

