/*
 * multi_threaded_udp_logger.h
 *
 *  Created on: Dec 7, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_UDP_LOGGING_COMMON_MULTI_THREADED_UDP_HANDLER_H_
#define INCLUDE_UDP_LOGGING_COMMON_MULTI_THREADED_UDP_HANDLER_H_

#include "f1_datalogger/udp_logging/f1_2018_datagrab_handler.h"
#include <tbb/concurrent_queue.h>
#include <tbb/task_group.h>
#include <memory>
#include <mutex>
#include <google/protobuf/util/json_util.h>
namespace deepf1
{

class MultiThreadedUDPHandler2018 : public IF12018DataGrabHandler
{
public:
  MultiThreadedUDPHandler2018(std::string data_folder = "udp_data", bool write_json = false, unsigned int sleeptime = 75);
  virtual ~MultiThreadedUDPHandler2018();
  inline bool isReady() override;
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override;
  const std::string getDataFolder() const;
  void stop();
  void join();
private:
  std::shared_ptr< tbb::task_group > thread_pool_ ;
  bool running_;
  deepf1::TimePoint begin_;
  const std::string data_folder_;

  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarSetupData> > setup_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarStatusData> > status_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData> > telemetry_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketEventData> > event_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketLapData> > lap_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketMotionData> > motion_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketParticipantsData> > participant_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketSessionData> > session_data_queue_;

  bool ready_;
  bool write_json_;
  unsigned int sleeptime_;
  google::protobuf::util::JsonOptions json_options_;

  void workerFunc(deepf1::twenty_eighteen::PacketID packetType);



  void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data) override;
  void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data) override;
  void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data) override;
  void handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data) override;
  void handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data) override;
  void handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data) override;
  void handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data) override;
  void handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) override;

};

} /* namespace deepf1 */

#endif /* INCLUDE_UDP_LOGGING_COMMON_MULTI_THREADED_UDP_HANDLER_H_ */
