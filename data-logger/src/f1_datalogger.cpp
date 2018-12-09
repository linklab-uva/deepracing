/*
 * f1_datalogger.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger.h"
#include "image_logging/utils/screencapture_lite_utils.h"
namespace deepf1
{

F1DataLogger::F1DataLogger(const std::string& search_string, std::shared_ptr<IF1FrameGrabHandler> frame_grab_handler, std::shared_ptr<IF1DatagrabHandler> data_grab_handler) :
    clock_(new std::chrono::high_resolution_clock())

{
  begin_ = clock_->now();
  std::cout<<"Creating managers"<<std::endl;
  frame_grab_manager_.reset(new F1FrameGrabManager(clock_, frame_grab_handler, search_string) );
  data_grab_manager_.reset(new F1DataGrabManager(clock_, data_grab_handler));
  std::cout<<"Created managers"<<std::endl;
  const scl::Window& window = frame_grab_manager_->getWindow();
  cv::Size size(window.Size.x, window.Size.y);
  frame_grab_handler->init(begin_, size);
  std::string host("127.0.0.1");
  data_grab_handler->init(host, 20777, begin_);

}
F1DataLogger::~F1DataLogger()
{
  data_grab_manager_->stop();
}

void F1DataLogger::start(double capture_frequency)
{
  frame_grab_manager_->start(capture_frequency);
  data_grab_manager_->start();
}

} /* namespace deepf1 */
