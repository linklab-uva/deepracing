/*
 * multi_threaded_capture.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */
#include "f1_datalogger.h"
#include "image_logging/common/multi_threaded_framegrab_handler.h"
#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace scl = SL::Screen_Capture;
class DummyUDPCaptureHandler : public deepf1::IF1DatagrabHandler
{
public:
  DummyUDPCaptureHandler()
  {

  }
  bool isReady() override
  {
    return true;
  }
  void handleData(const deepf1::TimestampedUDPData& data) override
  {
    deepf1::UDPPacket packet = data.data;
    printf("Got some data. Steering: %f. Throttle: %f. Brake: %f.\n", packet.m_steer, packet.m_throttle);
  }
  void init(const std::chrono::high_resolution_clock::time_point& begin) override
  {
    this->begin = begin;
  }
private:
  std::chrono::high_resolution_clock::time_point begin;
};
int main(int argc, char** argv)
{
  std::string search = "CMake";
  if (argc > 1)
  {
    search = std::string(argv[1]);
  }
  double capture_frequency = 10.0;
  if (argc > 2)
  {
    capture_frequency = atof(argv[2]);
  }
  std::shared_ptr<DummyUDPCaptureHandler> udp_handler(new DummyUDPCaptureHandler);
  std::shared_ptr<deepf1::MultiThreadedFrameGrabHandler> frame_handler(new deepf1::MultiThreadedFrameGrabHandler);
  deepf1::F1DataLogger dl(search, frame_handler, udp_handler);
  dl.start();

  std::this_thread::sleep_for(std::chrono::seconds(20));
}



