/*
 * multi_threaded_udp_logger.h
 *
 *  Created on: Dec 7, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_UDP_LOGGING_COMMON_MULTI_THREADED_UDP_HANDLER_H_
#define INCLUDE_UDP_LOGGING_COMMON_MULTI_THREADED_UDP_HANDLER_H_

#include "udp_logging/f1_datagrab_handler.h"
#include <tbb/concurrent_queue.h>
#include "tbb/task_group.h"
#include <memory>
#include <mutex>
namespace deepf1
{

class MultiThreadedUDPHandler : public IF1DatagrabHandler
{
public:
  MultiThreadedUDPHandler(std::string data_folder = "udp_data", unsigned int thread_count = 5);
  virtual ~MultiThreadedUDPHandler();
  void handleData(const deepf1::TimestampedUDPData& data) override;
  bool isReady() override;
  void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override;
  const std::string getDataFolder() const;

private:
  std::shared_ptr< tbb::concurrent_queue<TimestampedUDPData> > queue_;
  std::shared_ptr< tbb::task_group> thread_pool_ ;
  bool running_;
  std::chrono::high_resolution_clock::time_point begin_;
  unsigned int thread_count_;
  tbb::atomic<unsigned long> counter_;
  std::mutex queue_mutex_;
  const std::string data_folder_;

  void workerFunc_();

};

} /* namespace deepf1 */

#endif /* INCLUDE_UDP_LOGGING_COMMON_MULTI_THREADED_UDP_HANDLER_H_ */
