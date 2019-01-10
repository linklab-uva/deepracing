/*
 * multi_threaded_framegrab_handler.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */

#include "image_logging/common/multi_threaded_framegrab_handler.h"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include "TimestampedImage.pb.h"
#include <fstream>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
namespace deepf1
{

MultiThreadedFrameGrabHandler::MultiThreadedFrameGrabHandler(std::string images_folder, unsigned int thread_count) 
: running_(false), counter_(1), images_folder_(images_folder)
{
  fs::path imf(images_folder_);
  if(!fs::is_directory(imf))
  {
    fs::create_directories(imf);
  }
  thread_count_= thread_count;
}

MultiThreadedFrameGrabHandler::~MultiThreadedFrameGrabHandler()
{
  running_ = false;
  thread_pool_->cancel();
}

inline bool MultiThreadedFrameGrabHandler::isReady()
{
  return true;
}

void MultiThreadedFrameGrabHandler::handleData(const TimestampedImageData& data)
{
  std::lock_guard<std::mutex> lk(queue_mutex_);
  queue_->push(data);
}
void MultiThreadedFrameGrabHandler::init(const std::chrono::high_resolution_clock::time_point& begin,
                                         const cv::Size& window_size)
{
  begin_ = begin;
  running_ = true;
  queue_.reset(new tbb::concurrent_queue<TimestampedImageData>);
  thread_pool_.reset(new tbb::task_group);
  for(unsigned int i = 0; i < thread_count_; i ++)
  {
    thread_pool_->run(std::bind(&MultiThreadedFrameGrabHandler::workerFunc_,this));
  }
}

void MultiThreadedFrameGrabHandler::workerFunc_()
{
  std::cout<<"Spawned a worker thread to log images" <<std::endl;
  while(running_)
  {
    TimestampedImageData data;
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      if(queue_->empty())
      {
        continue;
      }
      if(!queue_->try_pop(data))
      {
        continue;
      }
    }
    unsigned long counter = counter_.fetch_and_increment();
    fs::path  images_folder(images_folder_);
    google::protobuf::uint64 delta = (google::protobuf::uint64)(std::chrono::duration_cast<std::chrono::microseconds>(data.timestamp - begin_).count());

    fs::path image_file("image_" + std::to_string(counter) + ".jpg");
    std::string fn = (images_folder / image_file).string();
    cv::imwrite(fn,data.image);


    deepf1::protobuf::TimestampedImage tag;
    tag.set_image_file(fn);
    tag.set_timestamp(delta);
    fs::path pb_file("image_" + std::to_string(counter) + ".pb");

    std::string pb_fn = (images_folder / pb_file).string();
    std::ofstream ostream(pb_fn.c_str());
    tag.SerializeToOstream(&ostream);
    ostream.close();
  }
}

} /* namespace deepf1 */
