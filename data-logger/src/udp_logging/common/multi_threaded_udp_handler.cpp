/*
 * multi_threaded_udp_logger.cpp
 *
 *  Created on: Dec 7, 2018
 *      Author: ttw2xk
 */

#include <udp_logging/common/multi_threaded_udp_handler.h>
#include <functional>
#include <boost/filesystem.hpp>
#include <iostream>
#include "F1UDPData.pb.h"
#include <thread>
namespace fs = boost::filesystem;
namespace deepf1
{
MultiThreadedUDPHandler::MultiThreadedUDPHandler(std::string data_folder, unsigned int thread_count) : running_(false), counter_(1), thread_count_(thread_count)
{
  fs::path df(data_folder);
  if(not fs::is_directory(df))
  {
    fs::create_directories(df);
  }
}
MultiThreadedUDPHandler::~MultiThreadedUDPHandler()
{
}

void MultiThreadedUDPHandler::handleData(const deepf1::TimestampedUDPData& data)
{
  std::lock_guard<std::mutex> lk(queue_mutex_);
  queue_->push(data);
}

bool MultiThreadedUDPHandler::isReady()
{
  return true;
}
void MultiThreadedUDPHandler::workerFunc_()
{

  std::cout<<"Spawned a worker thread to log udp data" <<std::endl;
  while(running_)
  {
    if(queue_->empty())
    {
      continue;
    }
    TimestampedUDPData data;
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      if(!queue_->try_pop(data))
      {
        continue;
      }
    }
    unsigned long counter = counter_.fetch_and_increment();
    fs::path  udp_folder("udp_data");
    google::protobuf::uint64 delta = (google::protobuf::uint64)(std::chrono::duration_cast<std::chrono::microseconds>(data.timestamp - begin_).count());



    deepf1::protobuf::F1UDPData udp_pb;
    udp_pb.set_brake(data.data.m_brake);
    udp_pb.set_logger_time(delta);
    udp_pb.set_game_lap_time(data.data.m_lapTime);
    udp_pb.set_game_time(data.data.m_time);
    udp_pb.set_steering(data.data.m_steer);
    udp_pb.set_throttle(data.data.m_throttle);
    fs::path pb_file("udp_packet_" + std::to_string(counter) + ".pb");
    std::string pb_fn = (udp_folder / pb_file).string();
    std::ofstream ostream(pb_fn.c_str());
    udp_pb.SerializeToOstream(&ostream);
    ostream.close();
  }
}
void MultiThreadedUDPHandler::init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin)
{
  begin_ = begin;
  running_ = true;
  queue_.reset(new tbb::concurrent_queue<TimestampedUDPData>);
  thread_pool_.reset(new tbb::task_group);
  for(int i = 0; i < thread_count_; i ++)
  {
    thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler::workerFunc_,this));
  }
}

} /* namespace deepf1 */
