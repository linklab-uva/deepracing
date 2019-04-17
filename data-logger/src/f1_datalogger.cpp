/*
 * f1_datalogger.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/f1_datalogger.h"
//#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
namespace deepf1
{

F1DataLogger::F1DataLogger(const std::string& search_string, std::shared_ptr<IF1FrameGrabHandler> frame_grab_handler, std::shared_ptr<IF1DatagrabHandler> data_grab_handler,
	std::string host, unsigned int port) :
    clock_(new std::chrono::high_resolution_clock())

{
  begin_ = std::chrono::high_resolution_clock::time_point(clock_->now());
  std::cout<<"Creating managers"<<std::endl;
  std::cout << "Creating Data Grab Manager" << std::endl;
  data_grab_manager_.reset(new F1DataGrabManager(clock_, data_grab_handler, host, port));
  std::cout << "Created Data Grab Manager" << std::endl;
  std::cout << "Creating Frame Grab Manager" << std::endl;
  frame_grab_manager_.reset(new F1FrameGrabManager(clock_, frame_grab_handler, search_string) );
  std::cout << "Created Frame Grab Manager" << std::endl;
  std::cout<<"Created managers"<<std::endl;
  const scl::Window& window = frame_grab_manager_->getWindow();
  cv::Size size;
  size.height = window.Size.y;
  size.width = window.Size.x;

  std::cout<<"Got a Window of Size (W x H): " << std::endl << size << std::endl;
  frame_grab_handler->init(begin_, size);
  data_grab_handler->init(host, port, begin_);

}
F1DataLogger::~F1DataLogger()
{
  data_grab_manager_->stop();
  data_grab_manager_.reset();
  frame_grab_manager_.reset();
}

void F1DataLogger::start(double capture_frequency)
{
  frame_grab_manager_->start(capture_frequency);
  data_grab_manager_->start();
}

const std::chrono::high_resolution_clock::time_point F1DataLogger::getStart() const
{
	return begin_;
}

} /* namespace deepf1 */
