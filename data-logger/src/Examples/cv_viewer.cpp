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
	/*  Eigen::Vector3d velocityGlobal(data.data.m_carMotionData[0].m_worldVelocityX, data.data.m_carMotionData[0].m_worldVelocityY, data.data.m_carMotionData[0].m_worldVelocityZ);
	  Eigen::Vector3d localVelocity(data.data.m_localVelocityX, data.data.m_localVelocityY, data.data.m_localVelocityZ);
	  std::cout << "Global Velocity: " << std::endl << velocityGlobal << std::endl;
	 
	  Eigen::Vector3d forward(data.data.m_carMotionData[0].m_worldForwardDirX, data.data.m_carMotionData[0].m_worldForwardDirY, data.data.m_carMotionData[0].m_worldForwardDirZ);
	  forward.normalize();
	  Eigen::Vector3d right(data.data.m_carMotionData[0].m_worldRightDirX, data.data.m_carMotionData[0].m_worldRightDirY, data.data.m_carMotionData[0].m_worldRightDirZ);
	  right.normalize();
	  Eigen::Vector3d up = right.cross(forward);
	  up.normalize();
	  Eigen::Matrix3d rotMatrix = Eigen::Matrix3d::Identity();
	  rotMatrix.col(0) = -right;
	  rotMatrix.col(1) = up;
	  rotMatrix.col(2) = forward;
	  Eigen::Quaterniond rotation(rotMatrix);

	  Eigen::Vector3d translation(data.data.m_carMotionData[0].m_worldPositionX, data.data.m_carMotionData[0].m_worldPositionY, data.data.m_carMotionData[0].m_worldPositionZ);
	  
	  Eigen::Vector3d velocityLocalComputed = rotation.inverse() * velocityGlobal;
	  Eigen::Vector3d velocityGlobalComputed = rotation * localVelocity;
	  std::cout << "Global Velocity Computed: " << std::endl << velocityGlobalComputed << std::endl;

	  Eigen::Vector3d eulerComputed = rotation.toRotationMatrix().eulerAngles(2, 1, 0);
	  Eigen::Vector3d euler(data.data.m_carMotionData[0].m_roll, data.data.m_carMotionData[0].m_pitch, data.data.m_carMotionData[0].m_yaw);

	  Eigen::Quaterniond rotationComputed(Eigen::AngleAxisd(eulerComputed[0], Eigen::Vector3d::UnitZ()) *
										  Eigen::AngleAxisd(eulerComputed[1], Eigen::Vector3d::UnitY()) *
										  Eigen::AngleAxisd(eulerComputed[2], Eigen::Vector3d::UnitX()));*/
	  //std::cout << "Rotation Delta: " << rotation.angularDistance(rotationComputed) << std::endl;
	  Eigen::Vector3d angularVelocityLocal(data.data.m_angularVelocityX, data.data.m_angularVelocityY, data.data.m_angularVelocityZ);
	 
	  std::cout << "Front Wheels Angle: " << data.data.m_frontWheelsAngle << std::endl;
	  std::cout << "angularVelocityLocal: " << std::endl << angularVelocityLocal << std::endl;
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
 //   cv::destroyWindow(window_name);
  }
  bool isReady() override
  {
    return true;
  }
  void handleData(const deepf1::TimestampedImageData& data) override
  {

  /*  long long delta = std::chrono::duration_cast<std::chrono::milliseconds>(data.timestamp - this->begin).count();
    std::stringstream ss;
    ss << delta << " milliseconds from start"<<std::endl;
	after = deepf1::TimePoint(data.timestamp);
	std::chrono::duration<double, std::ratio<1,1000> > deltat = (after - before);
	std::cout << "Delta t in milliseconds"
		<< deltat.count() << std::endl;
	before = deepf1::TimePoint(after);*/

 //   cv::Mat img_cv_video;
 //   cv::cvtColor(data.image, img_cv_video, cv::COLOR_BGRA2BGR);
	//cv::putText(img_cv_video, ss.str(), cv::Point(img_cv_video.cols/2 - ss.str().length(), img_cv_video.rows / 2), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0.0, 0.0, 0.0));
 //  // cv::imshow(window_name, data.image);
 //   video_writer_->write(img_cv_video);
  }
  void init(const deepf1::TimePoint& begin, const cv::Size& window_size) override
  {
   // cv::namedWindow(window_name);
  	//std::printf("Got a window of size: (W X H) (%d X %d)\n", window_size.width, window_size.height);
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
  std::cout<<"Enter anything to exit."<<std::endl;
  std::cin>>inp;
  dl.stop();
  std::cout << "Thanks for playing!" << std::endl;

}

