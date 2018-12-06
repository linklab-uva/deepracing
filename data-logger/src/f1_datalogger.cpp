/*
 * f1_datalogger.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger.h"
#include "image_logging/utils/screencapture_lite_utils.h"
namespace deepf1
{

F1DataLogger::F1DataLogger(const std::string& search_string, std::shared_ptr<IF1FrameGrabHandler> frame_grab_handler) :
    clock_(new std::chrono::high_resolution_clock())

{
  frame_grab_manager_.reset(new F1FrameGrabManager(search_string, clock_, frame_grab_handler) );

}
F1DataLogger::~F1DataLogger()
{
}

void F1DataLogger::start()
{
  frame_grab_manager_->start();
}

} /* namespace deepf1 */
