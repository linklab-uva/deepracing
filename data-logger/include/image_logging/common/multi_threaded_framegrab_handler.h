/*
 * multi_threaded_framegrab_handler.h
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_IMAGE_LOGGING_COMMON_MULTI_THREADED_FRAMEGRAB_HANDLER_H_
#define INCLUDE_IMAGE_LOGGING_COMMON_MULTI_THREADED_FRAMEGRAB_HANDLER_H_
#include "image_logging/framegrab_handler.h"
#include <tbb/concurrent_queue.h>
 #include "tbb/task_group.h"
namespace deepf1
{

class MultiThreadedFrameGrabHandler : public IF1FrameGrabHandler
{
public:
  MultiThreadedFrameGrabHandler(unsigned int thread_count = 5);
  virtual ~MultiThreadedFrameGrabHandler();
  bool isReady() override;
  void handleData(const TimestampedImageData& data) override;
  void init(const std::chrono::high_resolution_clock::time_point& begin, const cv::Size& window_size) override;
private:
  std::shared_ptr< tbb::concurrent_queue<TimestampedImageData> >queue_;
  std::shared_ptr< tbb::task_group> thread_pool_ ;
  bool running_;
  std::chrono::high_resolution_clock::time_point begin_;
  unsigned int thread_count_;

  void workerFunc_();
};

} /* namespace deepf1 */

#endif /* INCLUDE_IMAGE_LOGGING_COMMON_MULTI_THREADED_FRAMEGRAB_HANDLER_H_ */
