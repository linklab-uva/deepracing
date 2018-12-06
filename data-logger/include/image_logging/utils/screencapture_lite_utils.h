/*
 * screencapture_lite_utils.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_IMAGE_LOGGING_UTILS_SCREENCAPTURE_LITE_UTILS_H_
#define INCLUDE_IMAGE_LOGGING_UTILS_SCREENCAPTURE_LITE_UTILS_H_
#include "ScreenCapture.h"
namespace scl = SL::Screen_Capture;
namespace deepf1
{
class ScreencaptureLiteUtils
{
public:
  ScreencaptureLiteUtils();
  virtual ~ScreencaptureLiteUtils();
  static std::shared_ptr<scl::Window> findWindow(const std::string& search_string);
};

} /* namespace deepf1 */

#endif /* INCLUDE_IMAGE_LOGGING_UTILS_SCREENCAPTURE_LITE_UTILS_H_ */
