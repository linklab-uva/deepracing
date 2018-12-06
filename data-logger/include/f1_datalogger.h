/*
 * f1_datalogger.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_F1_DATALOGGER_H_
#define INCLUDE_F1_DATALOGGER_H_
#include "image_logging/f1_framegrab_manager.h"
namespace deepf1
{

class F1DataLogger
{
public:
  F1DataLogger(const std::string& search_string, std::shared_ptr<IF1FrameGrabHandler> frame_grab_handler);
  virtual ~F1DataLogger();

  void start();
private:
  std::shared_ptr<F1FrameGrabManager> frame_grab_manager_;

  std::shared_ptr<std::chrono::high_resolution_clock> clock_;
};

} /* namespace deepf1 */

#endif /* INCLUDE_F1_DATALOGGER_H_ */
