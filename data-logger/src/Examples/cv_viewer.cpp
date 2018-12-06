/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger.h"
#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>

namespace scl = SL::Screen_Capture;
class OpenCV_Viewer_Example_Handler : public deepf1::IF1FrameGrabHandler
{
public:
  OpenCV_Viewer_Example_Handler()
  {

  }
  virtual ~OpenCV_Viewer_Example_Handler()
  {

  }
  bool isReady() override
  {
    return true;
  }
  void handleData(const deepf1::TimestampedImageData& data) override
  {
    std::cout<<"Got some data"<<std::endl;
  }
};
scl::Window findWindow(const std::string& search_string)
{
  std::string srchterm(search_string);
  // convert to lower case for easier comparisons
  std::transform(srchterm.begin(), srchterm.end(), srchterm.begin(), [](char c)
  { return std::tolower(c, std::locale());});
  std::vector<scl::Window> filtereditems;

  std::vector<scl::Window> windows = SL::Screen_Capture::GetWindows(); // @suppress("Function cannot be resolved")
  std::cout<< "What window do you want to capture?\n" <<std::endl;
  for (unsigned int i = 0; i < windows.size(); i++)
  {
    scl::Window a = windows[i];
    std::string name = a.Name;
    std::transform(name.begin(), name.end(), name.begin(), [](char c)
    { return std::tolower(c, std::locale());});
    if (name.find(srchterm) != std::string::npos)
    {
      filtereditems.push_back(a);
    }
  }
  for (unsigned int i = 0; i < filtereditems.size(); i++)
  {
    scl::Window a = filtereditems[i];
    std::string name = a.Name;
    printf("Enter %d for %s\n", i, name.c_str());
  }

  std::string input;
  std::cin >> input;
  int selected_index;
  scl::Window selected_window;
  try
  {
    selected_index = atoi( (const char*) input.c_str() ); // @suppress("Invalid arguments") not sure why this is happening...
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
int main(int argc, char** argv)
{
  std::string search;
  if (argc > 1)
  {
    search = std::string(argv[1]);
  }
  else
  {
    search = "CMake";
  }
 // scl::Window window = findWindow(search);
  std::shared_ptr<OpenCV_Viewer_Example_Handler> handler(new OpenCV_Viewer_Example_Handler());
  deepf1::F1DataLogger dl(search, handler);
  dl.start();


}

