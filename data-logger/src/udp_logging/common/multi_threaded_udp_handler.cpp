/*
 * multi_threaded_udp_logger.cpp
 *
 *  Created on: Dec 7, 2018
 *      Author: ttw2xk
 */

#include <udp_logging/common/multi_threaded_udp_handler.h>
#include <functional>
namespace deepf1
{
MultiThreadedUDPHandler::MultiThreadedUDPHandler(unsigned int thread_count) : running_(false), counter_(1), thread_count_(thread_count)
{

}
MultiThreadedUDPHandler::~MultiThreadedUDPHandler()
{
}

void MultiThreadedUDPHandler::handleData(const deepf1::TimestampedUDPData& data)
{
}

bool MultiThreadedUDPHandler::isReady()
{
  return false;
}
void MultiThreadedUDPHandler::workerFunc_()
{

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
