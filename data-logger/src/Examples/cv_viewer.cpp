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
    deepf1::UDPPacket packet = data.data;
    printf("Got some data. Steering: %f. Throttle: %f. Brake: %f. Global Time: %f. Lap Time: %f. FIA Flags: %f. Is spectating: %d\n", packet.m_steer, packet.m_throttle, packet.m_brake, packet.m_time, packet.m_lapTime, packet.m_vehicleFIAFlags, packet.m_is_spectating);
    printf("X, Y, Z: %f %f %f\n", data.data.m_car_data[1].m_worldPosition[0], data.data.m_car_data[1].m_worldPosition[1],data.data.m_car_data[1].m_worldPosition[2]);

  }
  void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override
  {
    this->begin = begin;
  }
private:
  std::chrono::high_resolution_clock::time_point begin;
};
class OpenCV_Viewer_Example_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
  OpenCV_Viewer_Example_FrameGrabHandler() :
      window_name("cv_example")
  {
    cv::namedWindow(window_name);
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
    //cv::Size s = img_cv_video.size();
   // std::cout<<"Image is: " << s.height<< " X " << s.width << std::endl;
    video_writer_->write(img_cv_video);
  }
  void init(const std::chrono::high_resolution_clock::time_point& begin, const cv::Size& window_size) override
  {
    video_writer_.reset(new cv::VideoWriter("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, window_size));
    this->begin = begin;
  }
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
  double capture_frequency = 60.0;
  if (argc > 2)
  {
    capture_frequency = atof(argv[2]);
  }
  std::shared_ptr<OpenCV_Viewer_Example_FrameGrabHandler> image_handler(new OpenCV_Viewer_Example_FrameGrabHandler());
  std::shared_ptr<OpenCV_Viewer_Example_DataGrabHandler> udp_handler(new OpenCV_Viewer_Example_DataGrabHandler());
  deepf1::F1DataLogger dl(search, image_handler, udp_handler);
  dl.start(capture_frequency);

  cv::waitKey(0);

}

