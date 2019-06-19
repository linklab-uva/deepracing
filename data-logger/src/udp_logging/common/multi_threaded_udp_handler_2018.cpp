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
#include "f1_datalogger/proto/TimestampedPacketCarStatusData.pb.h"
#include "f1_datalogger/proto/TimestampedPacketCarSetupData.pb.h"
#include "f1_datalogger/proto/TimestampedPacketCarTelemetryData.pb.h"
#include "f1_datalogger/proto/TimestampedPacketEventData.pb.h"
#include "f1_datalogger/proto/TimestampedPacketLapData.pb.h"
#include "f1_datalogger/proto/TimestampedPacketMotionData.pb.h"
#include "f1_datalogger/proto/TimestampedPacketParticipantsData.pb.h"
#include "f1_datalogger/proto/TimestampedPacketSessionData.pb.h"

namespace fs = boost::filesystem;
namespace deepf1
{
MultiThreadedUDPHandler2018::MultiThreadedUDPHandler2018( std::string data_folder, bool write_json, unsigned int sleeptime )
 : running_(false), data_folder_(data_folder), write_json_(write_json), sleeptime_(sleeptime)
{
  fs::path main_dir(data_folder);
  if(!fs::is_directory(main_dir))
  {
    fs::create_directories(main_dir);
  }
  setup_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarSetupData>);
  status_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarStatusData>);
  telemetry_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData>);
  event_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketEventData>);
  lap_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketLapData>);
  motion_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketMotionData>);
  participant_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketParticipantsData>);
  session_data_queue_.reset(new tbb::concurrent_queue<deepf1::twenty_eighteen::TimestampedPacketSessionData>);
  json_options_.add_whitespace = true;
  json_options_.always_print_primitive_fields = true;
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


void MultiThreadedUDPHandler2018::workerFunc(deepf1::twenty_eighteen::PacketID packetType)
{
  fs::path output_dir;
  tbb::atomic<unsigned long> counter(1);
  switch(packetType)
  {
    case deepf1::twenty_eighteen::PacketID::MOTION:
    {
      output_dir = fs::path(data_folder_) / fs::path("motion_packets");
      break;
    }
    case deepf1::twenty_eighteen::PacketID::SESSION:
    {
      output_dir = fs::path(data_folder_) / fs::path("session_packets");
      break;
    }
    case deepf1::twenty_eighteen::PacketID::LAPDATA:
    {
      output_dir = fs::path(data_folder_) / fs::path("lap_packets");
      break;
    }
    case deepf1::twenty_eighteen::PacketID::PARTICIPANTS:
    {
      output_dir = fs::path(data_folder_) / fs::path("participants_packets");
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARSETUPS:
    {
      output_dir = fs::path(data_folder_) / fs::path("car_setup_packets");
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARTELEMETRY:
    {
      output_dir = fs::path(data_folder_) / fs::path("car_telemetry_packets");
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARSTATUS:
    {
      output_dir = fs::path(data_folder_) / fs::path("car_status_packets");
      break;
    }
    default:
    {
      return;
    }
  }
  if(!fs::is_directory(output_dir))
  {
    fs::create_directories(output_dir);
  }

  using timeunit = std::chrono::milliseconds;
  std::unique_ptr<std::string> json_string( new std::string);
  std::ofstream ostream;
  switch(packetType)
  {
    case deepf1::twenty_eighteen::PacketID::MOTION:
    {
      deepf1::twenty_eighteen::TimestampedPacketMotionData timestamped_packet_f1;
      while(running_ || !(motion_data_queue_->empty()))
      {
        if(!motion_data_queue_->try_pop(timestamped_packet_f1))
        {
          continue;
        }
        deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData data_pb;
        data_pb.mutable_udp_packet()->CopyFrom(deepf1::twenty_eighteen::TwentyEighteenUDPStreamUtils::toProto(timestamped_packet_f1.data));
        google::protobuf::uint64 delta = (google::protobuf::uint64)(std::chrono::duration_cast<timeunit>(timestamped_packet_f1.timestamp - begin_).count());
        data_pb.set_timestamp(delta);
        fs::path filename = output_dir / fs::path("packet_" + std::to_string(counter.fetch_and_increment()) + ".json");
        google::protobuf::util::MessageToJsonString( data_pb, json_string.get(), json_options_ );
        ostream.open(filename.string().c_str());
        ostream << *json_string << std::endl;
        ostream.flush();
        ostream.close();
      }
      break;
    }
    case deepf1::twenty_eighteen::PacketID::SESSION:
    {
      break;
    }
    case deepf1::twenty_eighteen::PacketID::LAPDATA:
    {
      break;
    }
    case deepf1::twenty_eighteen::PacketID::PARTICIPANTS:
    {
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARSETUPS:
    {
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARTELEMETRY:
    {
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARSTATUS:
    {
      break;
    }
  }
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
  thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::CARSETUPS));
  thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::CARSTATUS));
  thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::CARTELEMETRY));
  thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::EVENT));
  thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::LAPDATA));
  thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::MOTION));
  thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::PARTICIPANTS));
  thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::SESSION));
}

const std::string MultiThreadedUDPHandler2018::getDataFolder() const
{
  return data_folder_;
}


} /* namespace deepf1 */
