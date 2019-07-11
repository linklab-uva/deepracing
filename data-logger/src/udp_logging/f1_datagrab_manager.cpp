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

F1DataGrabManager::F1DataGrabManager(std::shared_ptr<std::chrono::high_resolution_clock> clock,const std::string host,
                                     const unsigned int port, bool rebroadcast) :
    socket_(io_service_), rebroadcast_socket_(rebroadcast_io_context_), running_(true), rebroadcast_(rebroadcast)
{
  //socket_.set_option(boost::asio::ip::udp::socket::reuse_address(true));
  socket_.open(boost::asio::ip::udp::v4());

  if (rebroadcast_)
  {
    std::cout << "Openning rebroadcast socket " << std::endl;
    rebroadcast_socket_.open(boost::asio::ip::udp::v4());
    std::cout << "Openned rebroadcast socket " << std::endl;
  }
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
  clock_ = clock;
}
F1DataGrabManager::~F1DataGrabManager()
{
  running_ = false;
}
void F1DataGrabManager::run2017(std::shared_ptr<IF1DatagrabHandler> data_handler)
{
  //make space on the stack to receive packets.
  boost::system::error_code error;
  TimestampedUDPData data;
  boost::asio::mutable_buffer buff;
  while (running_)
  {
    std::size_t received_bytes = socket_.receive_from(boost::asio::buffer(&(data.data), BUFFER_SIZE), remote_endpoint_, 0, error);
    if (bool(data_handler) && data_handler->isReady())
    {
	  data.timestamp = clock_->now();
      data_handler->handleData(data);
    }
  }
}
void F1DataGrabManager::handle_send(boost::shared_ptr<std::string> message,
  const boost::system::error_code& error,
  std::size_t bytes_transferred)
{
  //std::printf("Rebroadcasted %zu bytes. Error code: %s\n", bytes_transferred, error.message().c_str());
}
void F1DataGrabManager::run2018(std::shared_ptr<IF12018DataGrabHandler> data_handler)
{
  boost::system::error_code error;
  char buffer[ F1DataGrabManager::BUFFER_SIZE ];
  deepf1::TimePoint timestamp;
  deepf1::twenty_eighteen::PacketHeader* header;
  boost::shared_ptr< std::string > send_message(new std::string("THIS IS A MESSAGE. YAY"));
  while (running_)
  {
    std::size_t received_bytes = socket_.receive_from(boost::asio::buffer(buffer, BUFFER_SIZE), remote_endpoint_, 0, error);
    timestamp = clock_->now();
    if (rebroadcast_)
    {
       rebroadcast_socket_.async_send_to(boost::asio::buffer(buffer, received_bytes), boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(socket_.local_endpoint().address().to_string()), socket_.local_endpoint().port() + 1), 0,
        boost::bind(&F1DataGrabManager::handle_send, this, send_message,
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred));
      rebroadcast_io_context_.run_one();
    }
    if (bool(data_handler) && data_handler->isReady())
    {
      header = reinterpret_cast<deepf1::twenty_eighteen::PacketHeader*>(buffer);

      /*
	    std::printf("Packet Type: %s. Number of bytes: %zu \n", packetIdMap.at(header->m_packetId).c_str(), received_bytes);
       *enum PacketID
        {
          MOTION=0,
          SESSION=1,
          LAPDATA=2,
          EVENT=3,
          PARTICIPANTS=4,
          CARSETUPS=5,
          CARTELEMETRY=6,
          CARSTATUS=7
        };
      */
      switch(header->m_packetId)
      {
        case deepf1::twenty_eighteen::PacketID::MOTION:
        {
          deepf1::twenty_eighteen::TimestampedPacketMotionData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketMotionData*>(buffer)), timestamp);
          data_handler->handleData(data);
          break;
        }
		    case deepf1::twenty_eighteen::PacketID::EVENT:
		    {
			    deepf1::twenty_eighteen::TimestampedPacketEventData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketEventData*>(buffer)), timestamp);
			    data_handler->handleData(data);
			    break;
		    }
        case deepf1::twenty_eighteen::PacketID::SESSION:
        {
          deepf1::twenty_eighteen::TimestampedPacketSessionData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketSessionData*>(buffer)), timestamp);
          data_handler->handleData(data);
          break;
        }
        case deepf1::twenty_eighteen::PacketID::LAPDATA:
        {
          deepf1::twenty_eighteen::TimestampedPacketLapData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketLapData*>(buffer)), timestamp);
          data_handler->handleData(data);
          break;
        }
        case deepf1::twenty_eighteen::PacketID::PARTICIPANTS:
        {
          deepf1::twenty_eighteen::TimestampedPacketParticipantsData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketParticipantsData*>(buffer)), timestamp);
          data_handler->handleData(data);
          break;
        }
        case deepf1::twenty_eighteen::PacketID::CARSETUPS:
        {
          deepf1::twenty_eighteen::TimestampedPacketCarSetupData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketCarSetupData*>(buffer)), timestamp);
          data_handler->handleData(data);
          break;
        }
        case deepf1::twenty_eighteen::PacketID::CARTELEMETRY:
        {
          deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketCarTelemetryData*>(buffer)), timestamp);
          data_handler->handleData(data);
          break;
        }
        case deepf1::twenty_eighteen::PacketID::CARSTATUS:
        {
          deepf1::twenty_eighteen::TimestampedPacketCarStatusData data(*(reinterpret_cast<deepf1::twenty_eighteen::PacketCarStatusData*>(buffer)), timestamp);
          data_handler->handleData(data);
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

void F1DataGrabManager::start(std::shared_ptr<IF12018DataGrabHandler> data_handler)
{
  run_thread_ = std::thread(std::bind(&F1DataGrabManager::run2018, this, data_handler));
}

void F1DataGrabManager::start(std::shared_ptr<IF1DatagrabHandler> data_handler)
{
  run_thread_ = std::thread(std::bind(&F1DataGrabManager::run2017, this, data_handler));
}

void F1DataGrabManager::stop()
{
	running_ = false;
	socket_.close();
}

} /* namespace deepf1 */
