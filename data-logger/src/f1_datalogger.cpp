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

F1DataLogger::F1DataLogger(const std::string& search_string, std::string host, unsigned int port) :
    clock_(new std::chrono::high_resolution_clock()), host_(host), port_(port)

{
  begin_ = std::chrono::high_resolution_clock::time_point(clock_->now());
  std::cout<<"Creating managers"<<std::endl;
	
	std::cout << "Creating Data Grab Manager" << std::endl;
  data_grab_manager_.reset(new F1DataGrabManager(clock_, host, port));
	std::cout << "Created Data Grab Manager" << std::endl;
  
	
	std::cout << "Creating Frame Grab Manager" << std::endl;
	frame_grab_manager_.reset(new F1FrameGrabManager(clock_, search_string));
	std::cout << "Created Frame Grab Manager" << std::endl;
  
  std::cout<<"Created managers"<<std::endl;
  
 


}
F1DataLogger::~F1DataLogger()
{
	stop();
	data_grab_manager_.reset();
	frame_grab_manager_.reset();
}

void F1DataLogger::countdown(unsigned int seconds, std::string txt)
{
	std::cout << txt << std::endl;
	for (int i = seconds; i > 0; i--)
	{
		std::cout << i << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

void F1DataLogger::stop()
{
	if (!(!frame_grab_manager_))
	{
		frame_grab_manager_->stop();
	}
	if (!(!data_grab_manager_))
	{
		data_grab_manager_->stop();
	}
}

void F1DataLogger::start(double capture_frequency, 
std::shared_ptr<IF12018DataGrabHandler> udp_handler, 
std::shared_ptr<IF1FrameGrabHandler> image_handler)
{
	if (bool(image_handler))
	{
	  const scl::Window& window = frame_grab_manager_->window_;
	  cv::Size size;
	  size.height = window.Size.y;
	  size.width = window.Size.x;
	  std::cout << "Got a Window of Size (W x H): " << std::endl << size << std::endl;
	  image_handler->init(begin_, size);
		frame_grab_manager_->start(capture_frequency, image_handler);
	}
	if (bool(udp_handler))
	{
			udp_handler->init(host_, port_, begin_);
	  	data_grab_manager_->start(udp_handler);
	}
}

void F1DataLogger::start(double capture_frequency, 
std::shared_ptr<IF1DatagrabHandler> udp_handler, 
std::shared_ptr<IF1FrameGrabHandler> image_handler)
{
	if (bool(udp_handler))
	{
			udp_handler->init(host_, port_, begin_);
	  	data_grab_manager_->start(udp_handler);
	}
	if (bool(image_handler))
	{
	  const scl::Window& window = frame_grab_manager_->window_;
	  cv::Size size;
	  size.height = window.Size.y;
	  size.width = window.Size.x;
	  std::cout << "Got a Window of Size (W x H): " << std::endl << size << std::endl;
	  image_handler->init(begin_, size);
		frame_grab_manager_->start(capture_frequency, image_handler);
	}
}

const deepf1::TimePoint F1DataLogger::getStart() const
{
	return begin_;
}

} /* namespace deepf1 */
