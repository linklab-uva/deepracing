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
#include "Win32WindowEnumeration.h"

namespace deepf1
{
deepf1::winrt_capture::Window selectWindow(const std::vector<deepf1::winrt_capture::Window>& filtereditems)
{
  for (int i = 0; i < filtereditems.size(); i++)
  {
    deepf1::winrt_capture::Window window = filtereditems[i];
    std::cout<<"Enter " << i << " for " << window.Title() << std::endl;
  }

  std::string input;
  std::cin >> input;
  int selected_index;
  try
  {
    selected_index = std::atoi( (const char*) input.c_str() );
    if (selected_index>=filtereditems.size())
    {
      throw std::out_of_range("");
    }
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
  return filtereditems.at(selected_index);
}
F1FrameGrabManager::F1FrameGrabManager(const deepf1::TimePoint& begin, const std::string& search_string)
{
  auto d3dDevice = CreateD3DDevice();
  auto dxgiDevice = d3dDevice.as<IDXGIDevice>();
  m_device = CreateDirect3DDevice(dxgiDevice.get());
  begin_ = deepf1::TimePoint(begin);
  std::cout << "Looking for an application with the search string " << search_string << std::endl;
  std::vector<deepf1::winrt_capture::Window> g_windows = EnumerateWindows();
  std::vector<deepf1::winrt_capture::Window> filtered_windows;
  std::for_each(g_windows.begin(), g_windows.end(),[&filtered_windows,search_string](const deepf1::winrt_capture::Window &window)
    {
      if (window.Title().find(search_string) != std::string::npos)
      {
        filtered_windows.push_back(window);
      }
    }
  );
  
  selected_window = std::make_shared<deepf1::winrt_capture::Window>( selectWindow(filtered_windows) );
  window_rows_ = selected_window->rows();
  window_cols_ = selected_window->cols();
  std::cout << "Stored the selected window " << search_string << std::endl;
}
F1FrameGrabManager::~F1FrameGrabManager()
{
	stop();
}

void F1FrameGrabManager::start(double capture_frequency, std::shared_ptr<IF1FrameGrabHandler> capture_handler)
{

  std::cout<<"Entering F1FrameGrabManager::start"<<std::endl;
  auto item = CreateCaptureItemForWindow(selected_window->Hwnd());
  std::cout<<"Creating WinrtCapture"<<std::endl;
  cap.reset(new WinrtCapture(m_device, item, capture_handler));
  std::cout<<"Starting capture"<<std::endl;
  cap->StartCapture();
  std::cout<<"Started capture"<<std::endl;
}
void F1FrameGrabManager::stop()
{
//	capture_manager_.reset
}

} /* namespace deepf1 */
