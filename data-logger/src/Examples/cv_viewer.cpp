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
#include <Eigen/Core>
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
	  std::printf("Got a telemetry packet. Steering Ratio: %d\n", data.data.m_carTelemetryData[car_index].m_steer);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data) override
  {
	  std::printf("Got an event packet %s.\n", (data.data.m_eventStringCode));
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data) override
  {
	 // std::printf("Got a motion data packet. Session timestamp: %f\n", data.data.m_header.m_sessionTime);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) override
  {
    car_index = data.data.m_spectatorCarIndex;
    
  }
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override
  {
    this->begin = begin;
  }
private:
  std::chrono::high_resolution_clock::time_point begin;
  uint8_t car_index;
  float t1 = 0.0;
  float t2 = 0.0;
};
class OpenCV_Viewer_Example_DataGrabHandler : public deepf1::IF1DatagrabHandler
{
public:
  OpenCV_Viewer_Example_DataGrabHandler()
  {

  }
  bool isReady() override
  {
    return true;
  }
  void handleData(const deepf1::TimestampedUDPData& data) override
  {
    deepf1::UDPPacket2017 packet = data.data;
    car_index = data.data.m_spectator_car_index;
	/*t2 = packet.m_lapTime;
	float deltat = t2 - t1;
	printf("t1, t2, dt: %f %f %f\n", t1, t2, deltat);
	t1 = t2;*/

  //  printf("Got some data. Steering: %f. Throttle: %f. Brake: %f. Global Time: %f. Lap Time: %f. FIA Flags: %f. Is spectating: %d\n", packet.m_steer, packet.m_throttle, packet.m_brake, packet.m_time, packet.m_lapTime, packet.m_vehicleFIAFlags, packet.m_is_spectating);
    printf("Car lap time %f\n", data.data.m_car_data[car_index].m_currentLapTime);

  }
  void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override
  {
    this->begin = begin;
  }
private:
  uint8_t car_index;
  std::chrono::high_resolution_clock::time_point begin;
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

    long long delta = std::chrono::duration_cast<std::chrono::milliseconds>(data.timestamp - this->begin).count();
    std::stringstream ss;
    ss << delta << " milliseconds from start"<<std::endl;
	after = deepf1::TimePoint(data.timestamp);
	std::chrono::duration<double, std::ratio<1,1000> > deltat = (after - before);
	std::cout << "Delta t in milliseconds"
		<< deltat.count() << std::endl;
	before = deepf1::TimePoint(after);

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

