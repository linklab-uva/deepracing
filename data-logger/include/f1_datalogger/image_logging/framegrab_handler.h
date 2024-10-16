/*
 * framegrab_handler.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_IMAGE_LOGGING_FRAMEGRAB_HANDLER_H_
#define INCLUDE_IMAGE_LOGGING_FRAMEGRAB_HANDLER_H_
#include "f1_datalogger/car_data/timestamped_image_data.h"
#include "f1_datalogger/image_logging/visibility_control.h"

namespace deepf1
{

class F1_DATALOGGER_IMAGE_LOGGING_PUBLIC IF1FrameGrabHandler
{
public:
  IF1FrameGrabHandler() = default;
  virtual ~IF1FrameGrabHandler() = default;
  virtual bool isReady() = 0;
  virtual void handleData(const TimestampedImageData& data) = 0;
  virtual void init(const std::chrono::high_resolution_clock::time_point& begin, const cv::Size& window_size) = 0;

};

} /* namespace deepf1 */

#endif /* INCLUDE_IMAGE_LOGGING_FRAMEGRAB_HANDLER_H_ */
