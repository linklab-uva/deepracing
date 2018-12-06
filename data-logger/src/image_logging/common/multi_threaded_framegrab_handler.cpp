/*
 * multi_threaded_framegrab_handler.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */

#include "image_logging/common/multi_threaded_framegrab_handler.h"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
namespace deepf1
{

MultiThreadedFrameGrabHandler::MultiThreadedFrameGrabHandler(unsigned int thread_count) : running_(false)
{
  thread_count_= thread_count;
}

MultiThreadedFrameGrabHandler::~MultiThreadedFrameGrabHandler()
{
  running_ = false;
}

bool MultiThreadedFrameGrabHandler::isReady()
{
  return true;
}

void MultiThreadedFrameGrabHandler::handleData(const TimestampedImageData& data)
{
///  std::cout<<"Handling data"<<std::endl;
//  TimestampedImageData data_copy;
//  data_copy.timestamp = data.timestamp;
//  data.image.copyTo(data_copy.image);
  queue_->push(data);

}
void MultiThreadedFrameGrabHandler::init(const std::chrono::high_resolution_clock::time_point& begin,
                                         const cv::Size& window_size)
{
  begin_ = begin;
  running_ = true;
  queue_.reset(new tbb::concurrent_queue<TimestampedImageData>);
  thread_pool_.reset(new tbb::task_group);
  for(int i = 0; i < thread_count_; i ++)
  {
    thread_pool_->run(std::bind(&MultiThreadedFrameGrabHandler::workerFunc_,this));
  }
}

void MultiThreadedFrameGrabHandler::workerFunc_()
{
  std::cout<<"Spawned a thread" <<std::endl;
  while(running_)
  {
    if(queue_->empty())
    {
   //   std::cout<<"Queue is empty"<<std::endl;
      continue;
    }
    TimestampedImageData data;
    if(!queue_->try_pop(data))
    {
      std::cout<<"Could not pop"<<std::endl;
      continue;
    }
    std::cout<<"Pop successful."<<std::endl;
    long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(data.timestamp - begin_).count();
    std::string fn = "image_" + std::to_string(milliseconds) + ".jpg";
    cv::imwrite(fn,data.image);
  }
}

} /* namespace deepf1 */
