/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger.h"
#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
namespace scl = SL::Screen_Capture;
class OpenCV_Viewer_Example_Handler : public deepf1::IF1FrameGrabHandler
{
public:
  OpenCV_Viewer_Example_Handler() : window_name("cv_example")
  {
    cv::namedWindow(window_name);
  }
  virtual ~OpenCV_Viewer_Example_Handler()
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
    std::stringstream ss;
    ss << delta << " milliseconds from start";

   // cv::putText(data.image, ss.str(), cv::Point(25,100), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0.0,0.0,0.0));
    ss << std::endl;
    printf("%s", ss.str().c_str());
    cv::imshow(window_name,data.image);
  }
  void init(const std::chrono::high_resolution_clock::time_point& begin) override
  {
    this->begin = begin;
  }
private:
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
  double capture_frequency = 15.0;
  if(argc>2)
  {
    capture_frequency = atof(argv[2]);
  }
  std::shared_ptr<OpenCV_Viewer_Example_Handler> handler(new OpenCV_Viewer_Example_Handler());
  deepf1::F1DataLogger dl(search, handler);
  dl.start();

  cv::waitKey(0);

}

