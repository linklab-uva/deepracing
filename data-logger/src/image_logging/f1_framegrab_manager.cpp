/*
 * F1FrameGrabManager.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/image_logging/f1_framegrab_manager.h"
#include "f1_datalogger/image_logging/utils/opencv_utils.h"
//#include "image_logging/utils/screencapture_lite_utils.h"
#include <algorithm>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

namespace scl = SL::Screen_Capture;
namespace deepf1
{
scl::Window findWindow(const std::string& search_string)
{
  std::string srchterm(search_string);
  // convert to lower case for easier comparisons
  std::transform(srchterm.begin(), srchterm.end(), srchterm.begin(), ::tolower);
  std::vector<scl::Window> filtereditems;

  std::vector<scl::Window> windows = SL::Screen_Capture::GetWindows(); // @suppress("Function cannot be resolved")
  for (unsigned int i = 0; i < windows.size(); i++)
  {
    scl::Window a = windows[i];
    std::string name = a.Name;
	//std::cout << name << std::endl;
	std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (name.find(srchterm) != std::string::npos)
    {
      filtereditems.push_back(a);
    }
  }
  for (int i = 0; i < filtereditems.size(); i++)
  {
    scl::Window a = filtereditems[i];
    std::string name(&(a.Name[0]));
    std::cout<<"Enter " << i << " for " << name << std::endl;
  }

  std::string input;
  std::cin >> input;
  int selected_index;
  scl::Window selected_window;
  try
  {
    selected_index = std::atoi( (const char*) input.c_str() );
    selected_window = filtereditems.at(selected_index);
  }
  catch (std::out_of_range &oor)
  {
    std::stringstream ss;
    ss << "Selected index (" << selected_index << ") is >= than the number of windows" << std::endl;
    ss << "Underlying exception message: " << std::endl << std::string(oor.what()) << std::endl;
    std::runtime_error ex(ss.str().c_str());
    throw ex;
  }
  catch (std::runtime_error &e)
  {
    std::stringstream ss;
    ss << "Could not grab selected window " << selected_index << std::endl;
    ss << "Underlying exception message " << std::string(e.what()) << std::endl;
    std::runtime_error ex(ss.str().c_str());
    throw ex;
  }
  return selected_window;
}
std::vector<scl::Monitor> monitorCB_()
{
  std::vector<scl::Monitor> allMon = scl::GetMonitors();
  std::vector<scl::Monitor> rtn; 
  for(const scl::Monitor& monitor : allMon)
  {
    if(monitor.Index==1)
    {
      rtn.push_back(monitor);
    }
  }
  return rtn;
}
F1FrameGrabManager::F1FrameGrabManager(const deepf1::TimePoint& begin, const std::string& search_string)
{
  begin_ = deepf1::TimePoint(begin);
  std::cout << "Looking for an application with the search string " << search_string << std::endl;
  window_ = findWindow(search_string);
  capture_config_ = scl::CreateCaptureConfiguration( (scl::WindowCallback)std::bind(&F1FrameGrabManager::get_windows_, this));
  capture_config_monitor_ = 
        scl::CreateCaptureConfiguration([]() {
            std::vector<scl::Monitor> allMonitors = scl::GetMonitors();
            std::vector<scl::Monitor> rtn; 
            for(const scl::Monitor& monitor : allMonitors)
            {
              if(monitor.Index==1)
              {
                rtn.push_back(monitor);
              }
            }
            return rtn;
        });
}
F1FrameGrabManager::~F1FrameGrabManager()
{
	stop();
}

std::vector<scl::Window> F1FrameGrabManager::get_windows_()
{
  return std::vector<scl::Window> {window_};
}
void F1FrameGrabManager::onNewFrame_(const scl::Image &img, const scl::Window &monitor, std::shared_ptr<IF1FrameGrabHandler> capture_handler)
{
  deepf1::TimePoint ts = deepf1::Clock::now();
  if (capture_handler->isReady())
  {
    TimestampedImageData timestamped_image(ts, deepf1::OpenCVUtils::toCV(img, monitor.Size));
    capture_handler->handleData(timestamped_image);
  }
}
void F1FrameGrabManager::onNewScreenFrame_(const scl::Image &img, const scl::Monitor &monitor, std::shared_ptr<IF1FrameGrabHandler> capture_handler)
{
  deepf1::TimePoint ts = deepf1::Clock::now();
  if (capture_handler->isReady())
  {
    TimestampedImageData timestamped_image(ts, deepf1::OpenCVUtils::toCV(img, scl::Point({monitor.Width, monitor.Height})));
    capture_handler->handleData(timestamped_image);
  }
}

void F1FrameGrabManager::stop()
{
//	capture_manager_.reset
	capture_config_.reset();
}
void F1FrameGrabManager::start(double capture_frequency, 
                    std::shared_ptr<IF1FrameGrabHandler> capture_handler)
{
  unsigned int ms = (unsigned int)(std::round(((double)1E3)/capture_frequency)); 
  capture_config_->onNewFrame((scl::WindowCaptureCallback)std::bind(&F1FrameGrabManager::onNewFrame_, this, std::placeholders::_1, std::placeholders::_2, capture_handler));
  capture_manager_ = capture_config_->start_capturing();
  //capture_config_monitor_->onNewFrame((scl::ScreenCaptureCallback)std::bind(&F1FrameGrabManager::onNewScreenFrame_, this, std::placeholders::_1, std::placeholders::_2, capture_handler));
  //capture_manager_ = capture_config_monitor_->start_capturing();
  capture_manager_->setFrameChangeInterval(std::chrono::milliseconds(ms));
  
  
}
} /* namespace deepf1 */
