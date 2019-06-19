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

namespace scl = SL::Screen_Capture;
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
	/*t2 = packet.m_lapTime;
	float deltat = t2 - t1;
	printf("t1, t2, dt: %f %f %f\n", t1, t2, deltat);
	t1 = t2;*/

  //  printf("Got some data. Steering: %f. Throttle: %f. Brake: %f. Global Time: %f. Lap Time: %f. FIA Flags: %f. Is spectating: %d\n", packet.m_steer, packet.m_throttle, packet.m_brake, packet.m_time, packet.m_lapTime, packet.m_vehicleFIAFlags, packet.m_is_spectating);
    printf("Car is at position %f, %f, %f\n", data.data.m_x, data.data.m_y, data.data.m_z);

  }
  void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override
  {
    this->begin = begin;
  }
private:
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
    cv::destroyWindow(window_name);
  }
  bool isReady() override
  {
    return true;
  }
  void handleData(const deepf1::TimestampedImageData& data) override
  {

    long long delta = std::chrono::duration_cast<std::chrono::nanoseconds>(data.timestamp - this->begin).count();
    //std::stringstream ss;
    //ss << delta << " milliseconds from start";

    // cv::putText(data.image, ss.str(), cv::Point(25,100), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0.0,0.0,0.0));
    cv::Mat img_cv_video;
    cv::cvtColor(data.image, img_cv_video, cv::COLOR_BGRA2BGR);
    cv::imshow(window_name, img_cv_video);
    video_writer_->write(img_cv_video);
  }
  void init(const std::chrono::high_resolution_clock::time_point& begin, const cv::Size& window_size) override
  {
  	cv::namedWindow(window_name);
    video_writer_.reset(new cv::VideoWriter("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), captureFreq, window_size));
    this->begin = begin;
  }
  static constexpr float captureFreq = 30.0;
private:
  std::shared_ptr<cv::VideoWriter> video_writer_;
  std::chrono::high_resolution_clock::time_point begin;
  std::string window_name;
};
int main(int argc, char** argv)
{
  std::string search = "CMake";
  if (argc > 1)
  {
    search = std::string(argv[1]);
  }
  std::shared_ptr<OpenCV_Viewer_Example_FrameGrabHandler> image_handler(new OpenCV_Viewer_Example_FrameGrabHandler());
  std::shared_ptr<OpenCV_Viewer_Example_DataGrabHandler> udp_handler(new OpenCV_Viewer_Example_DataGrabHandler());

  deepf1::F1DataLogger dl(search);
  // dl.start((double)OpenCV_Viewer_Example_FrameGrabHandler::captureFreq, udp_handler, std::shared_ptr<deepf1::IF1FrameGrabHandler>());

  // std::string inp;
  // std::cout<<"Enter anything to exit."<<std::endl;
  // std::cin>>inp;

  
  dl.start((double)OpenCV_Viewer_Example_FrameGrabHandler::captureFreq, udp_handler, image_handler);
  std::string inp;
  // std::cout<<"Enter anything to exit."<<std::endl;
  // std::cin>>inp;
  cv::waitKey(0);

}

