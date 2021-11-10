/*
 * F1FrameGrabManager.h
 *
 *  Created on: Dec 4, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_IMAGE_LOGGING_F1_FRAMEGRAB_MANAGER_H_
#define INCLUDE_IMAGE_LOGGING_F1_FRAMEGRAB_MANAGER_H_
#include "f1_datalogger/image_logging/framegrab_handler.h"
#include <chrono>
#include <f1_datalogger/image_logging/visibility_control.h>
#include <ScreenCapture.h>
namespace scl = SL::Screen_Capture;


namespace deepf1
{
class F1_DATALOGGER_IMAGE_LOGGING_PUBLIC F1FrameGrabManager
{

  friend class F1DataLogger;
public:
  F1FrameGrabManager(const deepf1::TimePoint& begin, const std::string& search_string = "F1");
  virtual ~F1FrameGrabManager();
private:
  void stop();
  void start(double capture_frequency, std::shared_ptr<IF1FrameGrabHandler> capture_handler);

  uint32_t window_rows_;
  uint32_t window_cols_;


  cv::Mat curr_image;
  deepf1::TimePoint begin_;

  #ifdef USE_WINRT_GRAPHICS
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_device{ nullptr };
    std::shared_ptr<OpencvCapture> cap;
    std::shared_ptr<deepf1::winrt_capture::Window> selected_window;
    std::shared_ptr<winrt::Windows::System::DispatcherQueueController> dqcontroller;
    std::shared_ptr<winrt::Windows::System::DispatcherQueue> dq;
  #else
    std::shared_ptr<scl::ICaptureConfiguration<scl::WindowCaptureCallback> > capture_config_;
    std::shared_ptr<scl::ICaptureConfiguration<scl::ScreenCaptureCallback> > capture_config_monitor_;
    std::shared_ptr<scl::IScreenCaptureManager> capture_manager_;
    void onNewFrame_(const scl::Image &img, const scl::Window &monitor, std::shared_ptr<IF1FrameGrabHandler> capture_handler);
    void onNewScreenFrame_(const scl::Image &img, const scl::Monitor &monitor, std::shared_ptr<IF1FrameGrabHandler> capture_handler);
  #endif



};

} /* namespace deepf1 */

#endif /* INCLUDE_IMAGE_LOGGING_F1_FRAMEGRAB_MANAGER_H_ */
