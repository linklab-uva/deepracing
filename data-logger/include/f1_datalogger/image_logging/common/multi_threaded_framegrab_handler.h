/*
 * multi_threaded_framegrab_handler.h
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_IMAGE_LOGGING_COMMON_MULTI_THREADED_FRAMEGRAB_HANDLER_H_
#define INCLUDE_IMAGE_LOGGING_COMMON_MULTI_THREADED_FRAMEGRAB_HANDLER_H_
#include "f1_datalogger/image_logging/framegrab_handler.h"
#include <tbb/concurrent_queue.h>
#include <tbb/task_group.h>
#include <mutex>
#include <memory>
#include <f1_datalogger/visibility_control.h>

namespace deepf1
{
struct MultiThreadedFrameGrabHandlerSettings
{
  public:
    MultiThreadedFrameGrabHandlerSettings()
    : image_extension("jpg"), images_folder("images"), thread_count(5), write_json(false), capture_region_ratio(1.0)
    {

    }
    std::string image_extension;
    std::string images_folder;
    unsigned int thread_count;
    bool write_json;
    double capture_region_ratio;

};
class F1_DATALOGGER_PUBLIC MultiThreadedFrameGrabHandler : public IF1FrameGrabHandler
{
	using timeunit = std::milli;
public:
  MultiThreadedFrameGrabHandler(MultiThreadedFrameGrabHandlerSettings settings = MultiThreadedFrameGrabHandlerSettings());
  virtual ~MultiThreadedFrameGrabHandler();
  inline bool isReady() override;
  void handleData(const TimestampedImageData& data) override;
  void init(const deepf1::TimePoint& begin, const cv::Size& window_size) override;
  const std::string getImagesFolder() const;
  void resume();
  void pause();
  void stop();
  void join();
private:
  std::shared_ptr< tbb::concurrent_queue<TimestampedImageData> >queue_;
  std::shared_ptr< tbb::task_group >  thread_pool_ ;
  bool running_;
  std::chrono::high_resolution_clock::time_point begin_;
  unsigned int thread_count_;
  tbb::atomic<unsigned long> counter_;
  std::mutex queue_mutex_;
  const std::string images_folder_;
  void workerFunc_();
  
  bool ready_;
  bool write_json_;

  const std::string image_extension_;
  double capture_region_ratio;
};

} /* namespace deepf1 */

#endif /* INCLUDE_IMAGE_LOGGING_COMMON_MULTI_THREADED_FRAMEGRAB_HANDLER_H_ */
