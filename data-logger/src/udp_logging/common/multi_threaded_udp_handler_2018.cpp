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
#include <sstream>
#include <exception>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "f1_datalogger/filesystem_helper.h"
namespace deepf1
{

MultiThreadedUDPHandler2018::MultiThreadedUDPHandler2018(MultiThreadedUDPHandler2018Settings settings)
 : running_(false), ready_(false), hard_stopped_(true), data_folder_(settings.udp_directory), 
    thread_settings_(settings), write_json_(settings.write_json), sleeptime_(settings.sleeptime), 
    setups_counter(1), status_counter(1), telemetry_counter(1), lapdata_counter(1),  motion_counter(1), participants_counter(1), session_counter(1), paused_(false)                             
{
  fs::path main_dir(data_folder_);
  if(fs::is_directory(main_dir))
  {
    std::string in("asdf");
    while (!(in.compare("y") == 0 || in.compare("n") == 0))
    {
      std::cout << "UDP Directory: " << main_dir.string() << " already exists. Overwrite it with new data? [y\\n]";
      std::cin >> in;
    }
    if (in.compare("y") == 0)
    {
      fs::remove_all(main_dir);
    }
    else
    {
      std::cout << "Thanks for playing!" << std::endl;
      exit(0); // @suppress("undefined")
    }
  }
  fs::create_directories(main_dir);
    
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
  hardStop();
}
void MultiThreadedUDPHandler2018::hardStop()
{
  running_ = false;
  ready_ = false;
  hard_stopped_ = true;
  thread_pool_->cancel();
}
void MultiThreadedUDPHandler2018::stop()
{
  std::cout<<"Cleaning up remaining udp packets"<<std::endl;
  std::printf("Motion Data Packets Remaining: %zu\n", motion_data_queue_->unsafe_size());
  std::printf("Car Setup Data Packets Remaining: %zu\n", setup_data_queue_->unsafe_size());
  std::printf("Car Status Data Packets Remaining: %zu\n", status_data_queue_->unsafe_size());
  std::printf("Car Telemetry Data Packets Remaining: %zu\n", telemetry_data_queue_->unsafe_size());
  std::printf("Lap Data Packets Remaining: %zu\n", lap_data_queue_->unsafe_size());
  std::printf("Participant Data Packets Remaining: %zu\n", participant_data_queue_->unsafe_size());
  std::printf("Session Data Packets Remaining: %zu\n", session_data_queue_->unsafe_size());
  running_ = false;
  ready_ = false;
}
void MultiThreadedUDPHandler2018::join(unsigned int extra_threads)
{ 
  for(unsigned int i = 0; i < extra_threads; i ++)
  {
    thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::CARSTATUS));
    thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::CARTELEMETRY));
    thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::LAPDATA));
    thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc,this, deepf1::twenty_eighteen::PacketID::MOTION));
  }
  thread_pool_->wait();
}

void MultiThreadedUDPHandler2018::addPausedFunction(const std::function<void()>& f)
{
  pausedHandlers_.push_back(f);
}

void MultiThreadedUDPHandler2018::addUnpausedFunction(const std::function<void()>& f)
{
  unpausedHandlers_.push_back(f);
}

inline bool MultiThreadedUDPHandler2018::isReady()
{
  return ready_;
}

template<class ProtoType, class F1Type, class timeunit>
inline void dispositionProto(ProtoType& data_pb, F1Type& timestamped_packet_f1, const unsigned int& sleeptime,
const deepf1::TimePoint& begin, const fs::path& output_dir,std::atomic<std::uint64_t>& counter, bool use_json)
{
  data_pb.mutable_udp_packet()->CopyFrom(deepf1::twenty_eighteen::TwentyEighteenUDPStreamUtils::toProto(timestamped_packet_f1.data));
  std::chrono::duration<double, timeunit> dt = timestamped_packet_f1.timestamp - begin;
  data_pb.set_timestamp(dt.count());
  if (use_json)
  {
    google::protobuf::util::JsonOptions json_options;
    json_options.add_whitespace = true;
    json_options.always_print_primitive_fields = true;
    std::string json_string;
    fs::path filename = output_dir / fs::path("packet_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + ".json");
    google::protobuf::util::Status rc = google::protobuf::util::MessageToJsonString(data_pb, &json_string, json_options);
    std::ofstream ostream(filename.string(), std::fstream::out | std::fstream::trunc);
    ostream << json_string << std::endl;
    ostream.flush();
    ostream.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(sleeptime));
  }
  else
  {
    fs::path filename = output_dir / fs::path("packet_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + ".pb");
    std::ofstream ostream(filename.string(), std::fstream::out | std::fstream::trunc | std::fstream::binary);
    data_pb.SerializeToOstream(&ostream);
    ostream.flush();
    ostream.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(sleeptime));
  }
}

void MultiThreadedUDPHandler2018::workerFunc(deepf1::twenty_eighteen::PacketID packetType)
{
  fs::path output_dir;
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
    case deepf1::twenty_eighteen::PacketID::EVENT:
    {
      output_dir = fs::path(data_folder_) / fs::path("event_packets");
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
  switch(packetType)
  {
    case deepf1::twenty_eighteen::PacketID::MOTION:
    {
      deepf1::twenty_eighteen::TimestampedPacketMotionData timestamped_packet_f1;
      while(!hard_stopped_ && (running_ || !(motion_data_queue_->empty())))
      {
        if(!motion_data_queue_->try_pop(timestamped_packet_f1))
        {
          continue;
        }
        deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData data_pb;
        dispositionProto<deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData, deepf1::twenty_eighteen::TimestampedPacketMotionData, timeunit>
        (data_pb, timestamped_packet_f1, sleeptime_, begin_, output_dir, motion_counter, write_json_);
      }
      break;
    }
    case deepf1::twenty_eighteen::PacketID::SESSION:
    {
      deepf1::twenty_eighteen::TimestampedPacketSessionData timestamped_packet_f1;
      while(!hard_stopped_ && (running_ || !(session_data_queue_->empty())))
      {
        if(!session_data_queue_->try_pop(timestamped_packet_f1))
        {
          continue;
        }
        deepf1::twenty_eighteen::protobuf::TimestampedPacketSessionData data_pb;
        dispositionProto<deepf1::twenty_eighteen::protobuf::TimestampedPacketSessionData, deepf1::twenty_eighteen::TimestampedPacketSessionData, timeunit>
        (data_pb, timestamped_packet_f1, sleeptime_, begin_, output_dir, session_counter, write_json_);
      }
      break;
    }
    case deepf1::twenty_eighteen::PacketID::LAPDATA:
    {
      deepf1::twenty_eighteen::TimestampedPacketLapData timestamped_packet_f1;
      while(!hard_stopped_ && (running_ || !(lap_data_queue_->empty())))
      {
        if(!lap_data_queue_->try_pop(timestamped_packet_f1))
        {
          continue;
        }
        deepf1::twenty_eighteen::protobuf::TimestampedPacketLapData data_pb;
        dispositionProto<deepf1::twenty_eighteen::protobuf::TimestampedPacketLapData, deepf1::twenty_eighteen::TimestampedPacketLapData, timeunit>
        (data_pb, timestamped_packet_f1, sleeptime_, begin_, output_dir, lapdata_counter, write_json_);
      }
      break;
    }
    case deepf1::twenty_eighteen::PacketID::EVENT:
    {
      break;
    }
    case deepf1::twenty_eighteen::PacketID::PARTICIPANTS:
    {
      deepf1::twenty_eighteen::TimestampedPacketParticipantsData timestamped_packet_f1;
      while(!hard_stopped_ && (running_ || !(participant_data_queue_->empty())))
      {
        if(!participant_data_queue_->try_pop(timestamped_packet_f1))
        {
          continue;
        }
        deepf1::twenty_eighteen::protobuf::TimestampedPacketParticipantsData data_pb;
        dispositionProto<deepf1::twenty_eighteen::protobuf::TimestampedPacketParticipantsData, deepf1::twenty_eighteen::TimestampedPacketParticipantsData, timeunit>
        (data_pb, timestamped_packet_f1, sleeptime_, begin_, output_dir, participants_counter, write_json_);
      }
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARSETUPS:
    {
      deepf1::twenty_eighteen::TimestampedPacketCarSetupData timestamped_packet_f1;
      while(!hard_stopped_ && (running_ || !(setup_data_queue_->empty())))
      {
        if(!setup_data_queue_->try_pop(timestamped_packet_f1))
        {
          continue;
        }
        deepf1::twenty_eighteen::protobuf::TimestampedPacketCarSetupData data_pb;
        dispositionProto<deepf1::twenty_eighteen::protobuf::TimestampedPacketCarSetupData, deepf1::twenty_eighteen::TimestampedPacketCarSetupData, timeunit>
        (data_pb, timestamped_packet_f1, sleeptime_, begin_, output_dir, setups_counter, write_json_);
      }
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARTELEMETRY:
    {
      deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData timestamped_packet_f1;
      while(!hard_stopped_ && (running_ || !(telemetry_data_queue_->empty())))
      {
        if(!telemetry_data_queue_->try_pop(timestamped_packet_f1))
        {
          continue;
        }
        deepf1::twenty_eighteen::protobuf::TimestampedPacketCarTelemetryData data_pb;
        dispositionProto<deepf1::twenty_eighteen::protobuf::TimestampedPacketCarTelemetryData, deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData, timeunit>
        (data_pb, timestamped_packet_f1, sleeptime_, begin_, output_dir, telemetry_counter, write_json_);
      }
      break;
    }
    case deepf1::twenty_eighteen::PacketID::CARSTATUS:
    {
      deepf1::twenty_eighteen::TimestampedPacketCarStatusData timestamped_packet_f1;
      while(!hard_stopped_ && (running_ || !(status_data_queue_->empty())))
      {
        if(!status_data_queue_->try_pop(timestamped_packet_f1))
        {
          continue;
        }
        deepf1::twenty_eighteen::protobuf::TimestampedPacketCarStatusData data_pb;
        dispositionProto<deepf1::twenty_eighteen::protobuf::TimestampedPacketCarStatusData, deepf1::twenty_eighteen::TimestampedPacketCarStatusData, timeunit>
        (data_pb, timestamped_packet_f1, sleeptime_, begin_, output_dir, status_counter, write_json_);
      }
      break;
    }
  }
}




void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data)
{
  if (thread_settings_.carsetupsThreads>0)
  {
    setup_data_queue_->push(data);
  }  
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data)
{
  if (thread_settings_.carsetupsThreads>0)
  {
    status_data_queue_->push(data);
  }  
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data)
{
  if (thread_settings_.cartelemetryThreads>0)
  {
    telemetry_data_queue_->push(data);
  }  
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data)
{
	std::string eventString((char*)data.data.m_eventStringCode, 4);
	std::printf("Got an event packet, %s", eventString.c_str());
	event_data_queue_->push(data);
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data)
{
  if (thread_settings_.lapDataThreads>0)
  {
    lap_data_queue_->push(data);
  }  
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data)
{
  if (thread_settings_.motionThreads>0)
  {
    motion_data_queue_->push(data);
  }  
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data)
{
  if (thread_settings_.participantsThreads>0)
  {
    participant_data_queue_->push(data);
  }  
}
void MultiThreadedUDPHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) 
{
  if (thread_settings_.sessionThreads>0)
  {
    session_data_queue_->push(data);
  }  
    
  if (bool(data.data.m_gamePaused) && !pausedHandlers_.empty())
  {
    for (const std::function<void()> f: pausedHandlers_)
    {
      f();
    }
  }
  else if (!bool(data.data.m_gamePaused) && !unpausedHandlers_.empty())
  {
    for (const std::function<void()> f: unpausedHandlers_)
    {
      f();
    }
  }
  paused_ = bool(data.data.m_gamePaused);
}
void MultiThreadedUDPHandler2018::init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin)
{
  begin_ = deepf1::TimePoint(begin);
  thread_pool_.reset(new tbb::task_group);
  ready_ = true;
  running_ = true;
  hard_stopped_ = false;
  for ( unsigned int i = 0; i < thread_settings_.carsetupsThreads; i++ ) { thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc, this, deepf1::twenty_eighteen::PacketID::CARSETUPS)); }
  for ( unsigned int i = 0; i < thread_settings_.carstatusThreads; i++ ) { thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc, this, deepf1::twenty_eighteen::PacketID::CARSTATUS)); }
  for ( unsigned int i = 0; i < thread_settings_.cartelemetryThreads; i++ ) { thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc, this, deepf1::twenty_eighteen::PacketID::CARTELEMETRY)); }
  for ( unsigned int i = 0; i < thread_settings_.eventThreads; i++ ) { thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc, this, deepf1::twenty_eighteen::PacketID::EVENT)); }
  for ( unsigned int i = 0; i < thread_settings_.lapDataThreads; i++ ) { thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc, this, deepf1::twenty_eighteen::PacketID::LAPDATA)); }
  for ( unsigned int i = 0; i < thread_settings_.motionThreads; i++ ) { thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc, this, deepf1::twenty_eighteen::PacketID::MOTION)); }
  for ( unsigned int i = 0; i < thread_settings_.participantsThreads; i++ ) { thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc, this, deepf1::twenty_eighteen::PacketID::PARTICIPANTS)); }
  for ( unsigned int i = 0; i < thread_settings_.sessionThreads; i++ ) { thread_pool_->run(std::bind<void>(&MultiThreadedUDPHandler2018::workerFunc, this, deepf1::twenty_eighteen::PacketID::SESSION)); }
  
}

const std::string MultiThreadedUDPHandler2018::getDataFolder() const
{
  return data_folder_;
}


} /* namespace deepf1 */
