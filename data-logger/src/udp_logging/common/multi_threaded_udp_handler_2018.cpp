/*
 * multi_threaded_udp_logger.cpp
 *
 *  Created on: Dec 7, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/udp_logging/common/multi_threaded_udp_handler_2018.h"
#include "f1_datalogger/proto/TimestampedUDPData.pb.h"
#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"
#include <functional>
#include <boost/filesystem.hpp>
#include <iostream>
#include <google/protobuf/util/json_util.h>
#include <thread>

namespace fs = boost::filesystem;
namespace deepf1
{
MultiThreadedUDPHandler2018::MultiThreadedUDPHandler2018( std::string data_folder, bool write_json )
 : running_(false), data_folder_(data_folder), write_json_(write_json)
{
  fs::path df(data_folder_);
  if(!fs::is_directory(df))
  {
    fs::create_directories(df);
  }
  setup_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarSetupData>);
  status_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarStatusData>);
  telemetry_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData>);
  event_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketEventData>);
  lap_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketLapData>);
  motion_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketMotionData>);
  participant_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketParticipantsData>);
  session_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketSessionData>);
}
MultiThreadedUDPHandler2018::~MultiThreadedUDPHandler2018()
{
  stop();
  thread_pool_->cancel();
}
void MultiThreadedUDPHandler2018::stop()
{
  running_ = false;
  ready_ = false;
}
void MultiThreadedUDPHandler2018::join()
{ 
  thread_pool_->wait();
}

inline bool MultiThreadedUDPHandler2018::isReady()
{
  return ready_;
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data)
{
    setup_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data)
{
    status_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data)
{
    telemetry_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data)
{
    event_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data)
{
    lap_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data)
{
    motion_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data)
{
    participant_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) 
{
    session_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin)
{
  begin_ = deepf1::TimePoint(begin);
  thread_pool_.reset(new tbb::task_group);
  ready_ = true;
  running_ = true;
}

const std::string MultiThreadedUDPHandler2018::getDataFolder() const
{
  return data_folder_;
}


} /* namespace deepf1 */
