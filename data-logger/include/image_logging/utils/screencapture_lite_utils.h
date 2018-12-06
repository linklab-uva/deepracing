/*
 * screencapture_lite_utils.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_IMAGE_LOGGING_UTILS_SCREENCAPTURE_LITE_UTILS_H_
#define INCLUDE_IMAGE_LOGGING_UTILS_SCREENCAPTURE_LITE_UTILS_H_
#include "ScreenCapture.h"
#include <algorithm>
#include <iostream>
#include <sstream>
namespace scl = SL::Screen_Capture;
namespace deepf1
{
namespace utils
{
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
} /* namespace utils */

} /* namespace deepf1 */

#endif /* INCLUDE_IMAGE_LOGGING_UTILS_SCREENCAPTURE_LITE_UTILS_H_ */
