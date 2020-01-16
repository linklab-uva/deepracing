/*
 * multi_threaded_udp_logger.cpp
 *
 *  Created on: Dec 7, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/udp_logging/common/multi_threaded_udp_handler.h"
#include "f1_datalogger/proto/TimestampedUDPData.pb.h"
#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"
#include <functional>
#include <iostream>
#include <google/protobuf/util/json_util.h>
#include <thread>
#include <fstream>
#include <sstream>
#ifdef BOOST_FILESYSTEM
  #include <boost/filesystem.hpp>
  namespace fs = boost::filesystem;
#else
  #include <filesystem>
  namespace fs = std::filesystem;
#endif
namespace deepf1
{
MultiThreadedUDPHandler::MultiThreadedUDPHandler( std::string data_folder, unsigned int thread_count, bool write_json )
 : running_(false), counter_(1), thread_count_(thread_count), data_folder_(data_folder), write_json_(write_json)
{
  fs::path df(data_folder_);
  if(!fs::is_directory(df))
  {
    fs::create_directories(df);
  }
}
MultiThreadedUDPHandler::~MultiThreadedUDPHandler()
{
  stop();
  thread_pool_->cancel();
}
void MultiThreadedUDPHandler::stop()
{
  running_ = false;
  ready_ = false;
}
void MultiThreadedUDPHandler::join()
{
 
  {
    std::unique_lock<std::mutex> lk(queue_mutex_);
    printf("Cleaning up %ud remaining udp packets in the queue.\n", (unsigned int)queue_->unsafe_size());
  }
  thread_pool_->wait();
}
void MultiThreadedUDPHandler::handleData(const deepf1::TimestampedUDPData& data)
{
//  std::lock_guard<std::mutex> lk(queue_mutex_);
  queue_->push(data);
}

inline bool MultiThreadedUDPHandler::isReady()
{
  return ready_;
}
void MultiThreadedUDPHandler::workerFunc_()
{

  std::cout<<"Spawned a worker thread to log udp data" <<std::endl;
  while(running_ || !queue_->empty())
  {
    if(queue_->empty())
    {
      continue;
    }
    TimestampedUDPData data;
    {
     //std::lock_guard<std::mutex> lk(queue_mutex_);
      if(!queue_->try_pop(data))
      {
        continue;
      }
    }
    unsigned long counter = counter_.fetch_and_increment();
    fs::path  udp_folder(data_folder_);
    google::protobuf::uint64 delta = (google::protobuf::uint64)(std::chrono::duration_cast<std::chrono::milliseconds>(data.timestamp - begin_).count());
	  //std::cout << "Got some udp data. Clock Delta = " << delta << std::endl;
    std::unique_ptr<std::ofstream> ostream(new std::ofstream);

    deepf1::protobuf::TimestampedUDPData udp_pb;
    udp_pb.set_timestamp(delta);
    deepf1::protobuf::UDPData data_protobuf = deepf1::UDPStreamUtils::toProto(data.data);
    udp_pb.mutable_udp_packet()->CopyFrom(data_protobuf);
    std::string pb_file( "udp_packet_" + std::to_string(counter) + ".pb" );
    std::string pb_fn = ( udp_folder / fs::path(pb_file) ).string();
    ostream->open( pb_fn.c_str() , std::ofstream::out );
    udp_pb.SerializeToOstream(ostream.get());
    ostream->flush();
    ostream->close();

    if(write_json_)
    {
      std::unique_ptr<std::string> json( new std::string );
      google::protobuf::util::JsonOptions opshinz;
      opshinz.always_print_primitive_fields = true;
      opshinz.add_whitespace = true;
      google::protobuf::util::MessageToJsonString( udp_pb, json.get(), opshinz );
      std::string json_file = pb_file + ".json";
      std::string json_fn = ( udp_folder / fs::path(json_file) ).string();
      ostream->open(json_fn.c_str(), std::ofstream::out);
      ostream->write(json->c_str(),json->length());
      ostream->flush();
      ostream->close();
    }

    
  }
}
void MultiThreadedUDPHandler::init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin)
{
  begin_ = std::chrono::high_resolution_clock::time_point(begin);
  running_ = true;
  queue_.reset(new tbb::concurrent_queue<TimestampedUDPData>);
  thread_pool_.reset(new tbb::task_group);
  for(unsigned int i = 0; i < thread_count_; i ++)
  {
    thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler::workerFunc_,this));
  }
  ready_ = true;
}
const std::string MultiThreadedUDPHandler::getDataFolder() const
{
  return data_folder_;
}
} /* namespace deepf1 */
