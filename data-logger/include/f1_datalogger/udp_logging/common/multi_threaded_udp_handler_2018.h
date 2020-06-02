/*
 * multi_threaded_udp_logger.h
 *
 *  Created on: Dec 7, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_UDP_LOGGING_COMMON_MULTI_THREADED_UDP_HANDLER_2018_H_
#define INCLUDE_UDP_LOGGING_COMMON_MULTI_THREADED_UDP_HANDLER_2018_H_

#include "f1_datalogger/udp_logging/f1_2018_datagrab_handler.h"
#include <tbb/concurrent_queue.h>
#include <tbb/task_group.h>
#include <memory>
#include <mutex>
#include <google/protobuf/util/json_util.h>
#include <vector>
#include <functional>
namespace deepf1
{
struct MultiThreadedUDPHandler2018Settings
{
public:
  MultiThreadedUDPHandler2018Settings()
  {
    motionThreads = 3;
    sessionThreads = 1;
    lapDataThreads = 2;
    eventThreads = 0;
    participantsThreads = 1;
    carsetupsThreads = 1;
    cartelemetryThreads = 2;
    carstatusThreads = 2;
    write_json=false;
    sleeptime=75;
    udp_directory="udp_data";
  }
  uint32_t motionThreads;
  uint32_t sessionThreads;
  uint32_t lapDataThreads;
  uint32_t eventThreads;
  uint32_t participantsThreads;
  uint32_t carsetupsThreads;
  uint32_t cartelemetryThreads;
  uint32_t carstatusThreads;
  uint32_t totalThreads()
  {
    return motionThreads + sessionThreads + lapDataThreads + eventThreads + participantsThreads + carsetupsThreads + cartelemetryThreads + carstatusThreads;
  }
  bool write_json;
  unsigned int sleeptime;
  std::string udp_directory;
};
class F1_DATALOGGER_PUBLIC MultiThreadedUDPHandler2018 : public IF12018DataGrabHandler
{
  using timeunit = std::milli;
public:
  MultiThreadedUDPHandler2018(MultiThreadedUDPHandler2018Settings settings = MultiThreadedUDPHandler2018Settings());
  virtual ~MultiThreadedUDPHandler2018();
  inline bool isReady() override;
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override;
  const std::string getDataFolder() const;
  void stop();
  void hardStop();
  void join(unsigned int extra_threads = 1);
  void addPausedFunction(const std::function<void()>& f);
  void addUnpausedFunction(const std::function<void()>& f);
  void setSleepTime(const unsigned int& sleeptime)
  {
    sleeptime_ = sleeptime;
  }

private:
  std::shared_ptr< tbb::task_group > thread_pool_ ;
  bool running_;
  bool hard_stopped_;
  deepf1::TimePoint begin_;
  const std::string data_folder_;
  std::vector < std::function<void()> > pausedHandlers_;
  std::vector < std::function<void()> > unpausedHandlers_;
  bool paused_;


  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarSetupData> > setup_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarStatusData> > status_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData> > telemetry_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketEventData> > event_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketLapData> > lap_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketMotionData> > motion_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketParticipantsData> > participant_data_queue_;
  std::shared_ptr< tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketSessionData> > session_data_queue_;
  tbb::atomic<unsigned long> setups_counter,status_counter,telemetry_counter,lapdata_counter,
                              motion_counter,participants_counter,session_counter;

  bool ready_;
  bool write_json_;
  unsigned int sleeptime_;
  google::protobuf::util::JsonOptions json_options_;
  MultiThreadedUDPHandler2018Settings thread_settings_;

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
