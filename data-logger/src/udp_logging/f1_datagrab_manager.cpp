/*
 * f1_datagrab_manager.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/udp_logging/f1_datagrab_manager.h"
#include <iostream>
#include <functional>
#include <boost/bind.hpp>
namespace deepf1
{

F1DataGrabManager::F1DataGrabManager(const deepf1::TimePoint& begin,const std::string host,
                                     const unsigned int port) :
    socket_(io_service_), running_(true)
{
  begin_ = deepf1::TimePoint(begin);
  //socket_.set_option(boost::asio::ip::udp::socket::reuse_address(true));
  std::cout << "Opening UDP Socket " << std::endl;
  socket_.open(boost::asio::ip::udp::v4());
  std::cout << "Openned UDP Socket " << std::endl;

  if (host.compare("") == 0)
  {
	  std::cout << "Listening in broadcast mode on port " << port << std::endl;
	  socket_.set_option(boost::asio::socket_base::broadcast(true));
	  socket_.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port));
  }
  else
  {
	  std::cout << "Binding to host " <<host<<" on port " << port << std::endl;
	  socket_.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(host), port));
    std::cout << "Bound socket " << std::endl;
  }
}
F1DataGrabManager::~F1DataGrabManager()
{
  running_ = false;
}

void handle_send(const boost::system::error_code& error,
  std::size_t bytes_transferred)
{
  if(error.failed())
  {
   std::printf("Failed to rebroadcast %lu bytes. Error code: %d. Error Message: %s\n",
              bytes_transferred, error.value(), error.message().c_str());
  }
}
void handle_send(const std::string metadata,
  const boost::system::error_code& error,
  std::size_t bytes_transferred)
{
  if(error.failed())
  {
   std::printf("Failed to rebroadcast %lu bytes. Error code: %d. Error Message: %s\n",
              bytes_transferred, error.value(), error.message().c_str());
   std::printf("Message metadata: %s\n", metadata.c_str());
  }
}
void F1DataGrabManager::run()
{
  boost::system::error_code error;
  char buffer[ F1DataGrabManager::BUFFER_SIZE ];
  deepf1::uint16 packet_format_2018 = 2018;
  deepf1::uint16 packet_format_2019 = 2019;
  deepf1::uint16 packet_format_2020 = 2020;
  deepf1::TimePoint timestamp;
  deepf1::twenty_eighteen::PacketHeader* header;
  boost::asio::ip::udp::endpoint rebroadcastendpoint;

  while (running_)
  {
    std::size_t received_bytes = socket_.receive_from(boost::asio::buffer(buffer, BUFFER_SIZE), remote_endpoint_, 0, error);
    timestamp = deepf1::Clock::now();
    if(memcmp(buffer,(char *)&packet_format_2018, 2)==0)
    {
      header = reinterpret_cast<deepf1::twenty_eighteen::PacketHeader*>(buffer);
      
      if (!handlers2018.empty())
      {

        switch(header->m_packetId)
        {
          case deepf1::twenty_eighteen::PacketID::MOTION:
          {
            deepf1::twenty_eighteen::TimestampedPacketMotionData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketMotionData*>(buffer)), timestamp);
            std::for_each(handlers2018.begin(), handlers2018.end(), [data](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler) && data_handler->isReady()){data_handler->handleData(data);}});
            break;
          }
          case deepf1::twenty_eighteen::PacketID::EVENT:
          {
            deepf1::twenty_eighteen::TimestampedPacketEventData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketEventData*>(buffer)), timestamp);
            std::for_each(handlers2018.begin(), handlers2018.end(), [data](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler) && data_handler->isReady()){data_handler->handleData(data);}});
            break;
          }
          case deepf1::twenty_eighteen::PacketID::SESSION:
          {
            deepf1::twenty_eighteen::TimestampedPacketSessionData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketSessionData*>(buffer)), timestamp);
            std::for_each(handlers2018.begin(), handlers2018.end(), [data](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler) && data_handler->isReady()){data_handler->handleData(data);}});
            break;
          }
          case deepf1::twenty_eighteen::PacketID::LAPDATA:
          {
            deepf1::twenty_eighteen::TimestampedPacketLapData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketLapData*>(buffer)), timestamp);
            std::for_each(handlers2018.begin(), handlers2018.end(), [data](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler) && data_handler->isReady()){data_handler->handleData(data);}});
            break;
          }
          case deepf1::twenty_eighteen::PacketID::PARTICIPANTS:
          {
            deepf1::twenty_eighteen::TimestampedPacketParticipantsData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketParticipantsData*>(buffer)), timestamp);
            std::for_each(handlers2018.begin(), handlers2018.end(), [data](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler) && data_handler->isReady()){data_handler->handleData(data);}});
            break;
          }
          case deepf1::twenty_eighteen::PacketID::CARSETUPS:
          {
            deepf1::twenty_eighteen::TimestampedPacketCarSetupData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketCarSetupData*>(buffer)), timestamp);
            std::for_each(handlers2018.begin(), handlers2018.end(), [data](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler) && data_handler->isReady()){data_handler->handleData(data);}});
            break;
          }
          case deepf1::twenty_eighteen::PacketID::CARTELEMETRY:
          {
            deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketCarTelemetryData*>(buffer)), timestamp);
            std::for_each(handlers2018.begin(), handlers2018.end(), [data](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler) && data_handler->isReady()){data_handler->handleData(data);}});
            break;
          }
          case deepf1::twenty_eighteen::PacketID::CARSTATUS:
          {
            deepf1::twenty_eighteen::TimestampedPacketCarStatusData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketCarStatusData*>(buffer)), timestamp);
            std::for_each(handlers2018.begin(), handlers2018.end(), [data](std::shared_ptr<IF12018DataGrabHandler> data_handler){ if(bool(data_handler) && data_handler->isReady()){data_handler->handleData(data);}});
            break;
          }
          default:
          {
            break;
          }
        }
      }
    }
  }
}

void F1DataGrabManager::start()
{
  run_thread_ = std::thread(std::bind(&F1DataGrabManager::run, this));
}

void F1DataGrabManager::stop()
{
	running_ = false;
	socket_.close();
}

} /* namespace deepf1 */
