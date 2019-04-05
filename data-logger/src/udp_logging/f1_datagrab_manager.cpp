/*
 * f1_datagrab_manager.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/udp_logging/f1_datagrab_manager.h"
#include <iostream>
#include <functional>
namespace deepf1
{

F1DataGrabManager::F1DataGrabManager(std::shared_ptr<std::chrono::high_resolution_clock> clock,
                                     std::shared_ptr<IF1DatagrabHandler> handler, const std::string host,
                                     const unsigned int port) :
    socket_(io_service_), running_(true)
{
  socket_.open(boost::asio::ip::udp::v4());
  socket_.bind(boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(host), port));
  data_handler_ = handler;
  clock_ = clock;
}

F1DataGrabManager::~F1DataGrabManager()
{
  running_ = false;
}
void F1DataGrabManager::run_()
{
  unsigned int packet_size=1289;
  //make space on the stack to receive packets.
  boost::system::error_code error;
  char rcv_buffer[packet_size];
  while (running_)
  {
    std::size_t received_bytes = socket_.receive_from(boost::asio::buffer(rcv_buffer, packet_size), remote_endpoint_, 0, error);
    UDPPacket* fromChar = reinterpret_cast<UDPPacket*>(rcv_buffer);
    //std::cout<<"Got " << received_bytes << " bytes from the telemetry stream." << std::endl;
    if (!(!data_handler_) && data_handler_->isReady())
    {
      TimestampedUDPData data;
      data.data = *fromChar;
      data.timestamp = clock_->now();
      data_handler_->handleData(data);

    }
  }
}
void F1DataGrabManager::start()
{
  run_thread_ = std::thread(std::bind(&F1DataGrabManager::run_, this));
}
void F1DataGrabManager::stop()
{
	running_ = false;
	data_handler_.reset();
	socket_.close();
}
} /* namespace deepf1 */
