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
public:
  F1FrameGrabManager(std::shared_ptr<std::chrono::high_resolution_clock> clock,
                     std::shared_ptr<IF1FrameGrabHandler> capture_handler,
                     const std::string& search_string = "F12017");
  virtual ~F1FrameGrabManager();
  void stop();
  void start(double capture_frequency);

  const scl::Window& getWindow() const
  {
    return window_;
  }

private:
  std::shared_ptr<std::chrono::high_resolution_clock> clock_;

  scl::Window window_;

  std::shared_ptr<IF1FrameGrabHandler> capture_handler_;

  std::shared_ptr<scl::ICaptureConfiguration<scl::WindowCaptureCallback> > capture_config_;

  std::shared_ptr<scl::IScreenCaptureManager> capture_manager_;

  std::vector<scl::Window> get_windows_();

  void onNewFrame_(const scl::Image &img, const scl::Window &monitor);
};

} /* namespace deepf1 */

#endif /* INCLUDE_IMAGE_LOGGING_F1_FRAMEGRAB_MANAGER_H_ */
