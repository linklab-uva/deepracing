/*
 * multi_threaded_framegrab_handler.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/image_logging/common/multi_threaded_framegrab_handler.h"
#include "f1_datalogger/proto/TimestampedImage.pb.h"
#include <opencv2/imgcodecs.hpp>
#include <google/protobuf/util/json_util.h>
#include <thread>
#include <exception>
#include <filesystem>
#include <iostream>
#include <fstream>
namespace fs = std::filesystem;
namespace deepf1
{

MultiThreadedFrameGrabHandler::MultiThreadedFrameGrabHandler(std::string image_extension, std::string images_folder, unsigned int thread_count, bool write_json)
: running_(false), counter_(1), images_folder_(images_folder), write_json_(write_json), image_extension_(image_extension)
{
  fs::path main_dir(images_folder_);
  if(fs::is_directory(main_dir))
  {
    std::string in("asdf");
    while (!(in.compare("y") == 0 || in.compare("n") == 0))
    {
      std::cout << "Image Directory: " << main_dir.string() << " already exists. Overwrite it with new data? [y\\n]";
      std::cin >> in;
    }
    if ( in.compare("y") == 0 ) 
    {
      fs::remove_all(main_dir);
    }
    else
    {
      std::cout << "Thanks for playing!" << std::endl;
      exit(0);
    }
  }
  fs::create_directories(main_dir);
  thread_count_= thread_count;
}

MultiThreadedFrameGrabHandler::~MultiThreadedFrameGrabHandler()
{
  stop();
  thread_pool_->cancel();
}
void MultiThreadedFrameGrabHandler::join()
{
  {
    std::unique_lock<std::mutex> lk(queue_mutex_);
    printf("Cleaning up %ud remaining images in the queue.\n", (unsigned int)queue_->unsafe_size());
  }
  thread_pool_->wait();
}
void MultiThreadedFrameGrabHandler::resume()
{
 // 
  if (!ready_)
  {
    ready_ = true;
    std::cerr << "Resumed frame recording" << std::endl;
  }
}
void MultiThreadedFrameGrabHandler::pause()
{
  //  std::cerr << "Pausing frame recording" << std::endl;
  if (ready_)
  {
    ready_ = false;
    std::cerr << "Paused frame recording" << std::endl;
  }
}
void MultiThreadedFrameGrabHandler::stop()
{
  running_ = false;
  ready_ = false;
}
inline bool MultiThreadedFrameGrabHandler::isReady()
{
  return ready_;
}

void MultiThreadedFrameGrabHandler::handleData(const TimestampedImageData& data)
{
//  std::lock_guard<std::mutex> lk(queue_mutex_);
  queue_->push(data);
}
void MultiThreadedFrameGrabHandler::init(const deepf1::TimePoint& begin,
                                         const cv::Size& window_size)
{
  begin_ = std::chrono::high_resolution_clock::time_point(begin);
  running_ = true;
  queue_.reset(new tbb::concurrent_queue<TimestampedImageData>);
  thread_pool_.reset(new tbb::task_group);
  for(unsigned int i = 0; i < thread_count_; i ++)
  {
    thread_pool_->run(std::bind(&MultiThreadedFrameGrabHandler::workerFunc_,this));
  }
  ready_ = true;
}

void MultiThreadedFrameGrabHandler::workerFunc_()
{
  std::cout<<"Spawned a worker thread to log images" <<std::endl;
  std::unique_ptr<std::ofstream> ostream(new std::ofstream);
  fs::path images_folder(images_folder_);
  while( running_ || !queue_->empty() )
  {
    TimestampedImageData data;
    {
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
	  //std::cout << "Got some image data. Clock Delta = " << delta << std::endl;
    std::string file_prefix = "image_" + std::to_string(counter);
    std::string image_file( file_prefix + "." + image_extension_);
    cv::imwrite( ( images_folder / fs::path(image_file) ).string() , data.image );


    deepf1::protobuf::TimestampedImage tag;
    tag.set_image_file(image_file);
	  std::chrono::duration<double, timeunit> d = (data.timestamp - begin_);
    google::protobuf::uint64 delta = (google::protobuf::uint64)(std::round(d.count()));
    tag.set_timestamp(delta);

    if(write_json_)
    {
      std::unique_ptr<std::string> json( new std::string );
      google::protobuf::util::JsonOptions opshinz;
      opshinz.always_print_primitive_fields = true;
      opshinz.add_whitespace = true;
      google::protobuf::util::Status result = google::protobuf::util::MessageToJsonString( tag , json.get() , opshinz );
	  if (result.ok())
	  {
		  std::string json_filename(file_prefix + ".json");
		  std::string json_output_file((images_folder / fs::path(json_filename)).string());
		  ostream->open(json_output_file.c_str(), std::ofstream::out);
		  (*ostream) << (*json) << std::endl;
		  ostream->flush();
		  ostream->close();
	  }
    }
    else{
      std::string pb_filename( file_prefix + ".pb" );
      std::string pb_output_file(( images_folder / fs::path(pb_filename) ).string());
      ostream->open( pb_output_file.c_str(), std::ofstream::out );
      tag.SerializeToOstream( ostream.get() );
      ostream->flush();
      ostream->close();
    }
  }
}
const std::string MultiThreadedFrameGrabHandler::getImagesFolder() const
{
  return images_folder_;
}
} /* namespace deepf1 */
