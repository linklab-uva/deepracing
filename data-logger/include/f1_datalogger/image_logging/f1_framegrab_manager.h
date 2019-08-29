/*
 * F1FrameGrabManager.h
 *
 *  Created on: Dec 4, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_IMAGE_LOGGING_F1_FRAMEGRAB_MANAGER_H_
#define INCLUDE_IMAGE_LOGGING_F1_FRAMEGRAB_MANAGER_H_
#include "f1_datalogger/image_logging/framegrab_handler.h"
#include <ScreenCapture.h>
#include <chrono>
namespace deepf1
{
namespace scl = SL::Screen_Capture;
class F1FrameGrabManager
{

  friend class F1DataLogger;
public:
  F1FrameGrabManager(ClockPtr clock,
                     const std::string& search_string = "F1");
  virtual ~F1FrameGrabManager();
private:
  void stop();
  void start(double capture_frequency, 
                    std::shared_ptr<IF1FrameGrabHandler> capture_handler);



  ClockPtr clock_;

  scl::Window window_;

  std::shared_ptr<scl::ICaptureConfiguration<scl::WindowCaptureCallback> > capture_config_;
  std::shared_ptr<scl::ICaptureConfiguration<scl::ScreenCaptureCallback> > capture_config_monitor_;

  std::shared_ptr<scl::IScreenCaptureManager> capture_manager_;

  std::vector<scl::Window> get_windows_();

  void onNewFrame_(const scl::Image &img, const scl::Window &monitor, std::shared_ptr<IF1FrameGrabHandler> capture_handler);
  void onNewScreenFrame_(const scl::Image &img, const scl::Monitor &monitor, std::shared_ptr<IF1FrameGrabHandler> capture_handler);
};

} /* namespace deepf1 */

#endif /* INCLUDE_IMAGE_LOGGING_F1_FRAMEGRAB_MANAGER_H_ */
