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
    host_(host), port_(port), clock_(new std::chrono::high_resolution_clock()) 

{

  begin_ = deepf1::TimePoint(deepf1::Clock::now());//(clock_->now());
  std::cout<<"Creating managers"<<std::endl;
	
  std::cout << "Creating Data Grab Manager" << std::endl;
  data_grab_manager_.reset(new F1DataGrabManager(begin_, host, port));
  std::cout << "Created Data Grab Manager" << std::endl;
  
	
  std::cout << "Creating Frame Grab Manager" << std::endl;
  frame_grab_manager_.reset(new F1FrameGrabManager(begin_, search_string));
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
void F1DataLogger::add2018UDPHandler(std::shared_ptr<IF12018DataGrabHandler> udp_handler)
{
	data_grab_manager_->handlers2018.push_back(udp_handler);
}
void F1DataLogger::add2020UDPHandler(std::shared_ptr<IF12020DataGrabHandler> udp_handler)
{
	data_grab_manager_->handlers2020.push_back(udp_handler);
}

void F1DataLogger::start(double capture_frequency, 
std::shared_ptr<IF1FrameGrabHandler> image_handler)
{
	if (bool(image_handler))
	{
		cv::Size size(frame_grab_manager_->window_cols_, frame_grab_manager_->window_rows_);
		std::cout << "Got a Window of Size (W x H): " << std::endl << size << std::endl;
		image_handler->init(begin_, size);
		frame_grab_manager_->start(capture_frequency, image_handler);
	}
	if (!data_grab_manager_->handlers2018.empty())
	{
		std::for_each(data_grab_manager_->handlers2018.begin(), data_grab_manager_->handlers2018.end(), 
		[this](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler)){data_handler->init(host_,port_,begin_);}});
		data_grab_manager_->start();
	}
}

const deepf1::TimePoint F1DataLogger::getStart() const
{
	return begin_;
}

} /* namespace deepf1 */
