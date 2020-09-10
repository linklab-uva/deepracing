/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */
#include "f1_datalogger/f1_datalogger.h"
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
    // const int8_t& steer_value = data.data.m_carTelemetryData[car_index].m_steer;
    // const int8_t& throttle_value = data.data.m_carTelemetryData[car_index].m_throttle;
    // const int8_t& brake_value = data.data.m_carTelemetryData[car_index].m_brake;
	  // std::printf("Got a telemetry packet. Steering Ratio: %f\n", ((double)steer_value)/100.0);
	  // std::printf("Got a telemetry packet. Throttle Ratio: %f\n", ((double)throttle_value)/100.0);
	  // std::printf("Got a telemetry packet. Brake Ratio: %f\n", ((double)brake_value)/100.0);
	 // std::printf("Got a telemetry packet. Steering Ratio: %f\n", data.data.m_carTelemetryData[car_index].m_steer);
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
    const deepf1::twenty_eighteen::CarMotionData& car_data = data.data.m_carMotionData[car_index];
	  std::printf("Car position (X,Y,Z): (%f,%f,%f)\n", car_data.m_worldPositionX, car_data.m_worldPositionY, car_data.m_worldPositionZ);
    
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
    }
    ready = false;
    cv::Mat imcrop,imresize,imbgr;
    //imcrop = data.image;
    imcrop = data.image(cv::Range(0,(unsigned int)std::round((double)data.image.rows/scale_factor)),cv::Range(0,(unsigned int)std::round((double)data.image.cols/scale_factor)));
    cv::resize(imcrop,imresize,cv::Size(),1.0,1.0);
    if (!window_made)
    {
	    video_writer_.reset(new cv::VideoWriter("out.avi", cv::VideoWriter::fourcc('M','J','P','G'), captureFreq, cv::Size2i(imresize.cols, imresize.rows)));
      std::cout<<"Using video writer backend: " << video_writer_->getBackendName() << std::endl;
      window_made = true;
    }
    cv::cvtColor(imresize,imbgr, cv::COLOR_BGRA2BGR);
    video_writer_->write(imbgr);
    cv::imshow(window_name, imresize);
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
	  //video_writer_.reset(new cv::VideoWriter("out.avi", -1, captureFreq, window_size));
  }
  static constexpr float captureFreq = 30.0;
  const std::string window_name;
  double scale_factor;
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
  double scale_factor=1.0;
  if (argc > 1)
  {
    search = std::string(argv[1]);
  }
  if (argc > 2)
  {
    scale_factor = std::stod(std::string(argv[2]));
  }
  //std::shared_ptr<OpenCV_Viewer_Example_FrameGrabHandler> image_handler(new OpenCV_Viewer_Example_FrameGrabHandler());
  //image_handler->scale_factor=scale_factor;
  std::shared_ptr<deepf1::IF1FrameGrabHandler> image_handler;
  std::shared_ptr<OpenCV_Viewer_Example_2018DataGrabHandler> udp_handler(new OpenCV_Viewer_Example_2018DataGrabHandler());
  std::string inp;
  deepf1::F1DataLogger dl(search);  
  dl.start((double)OpenCV_Viewer_Example_FrameGrabHandler::captureFreq, udp_handler, image_handler);
  std::cout<<"Enter anything to exit."<<std::endl;
  std::string asdf;
  std::cin >> asdf;
 

}

