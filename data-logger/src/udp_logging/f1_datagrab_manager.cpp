/*
 * f1_datagrab_manager.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "udp_logging/f1_datagrab_manager.h"
#include <iostream>
#include <functional>
namespace deepf1
{

F1DataGrabManager::F1DataGrabManager(std::shared_ptr<IF1DatagrabHandler> handler, const std::string host, const unsigned int port) : socket_(io_service_), running_(true)
{
  socket_.open(udp::v4());
  socket_.bind(udp::endpoint(address::from_string(host), port));
  data_handler_ = handler;
  rcv_buffer_.reset(new UDPPacket);
}

F1DataGrabManager::~F1DataGrabManager()
{
  running_=false;
}
void F1DataGrabManager::run_()
{
  std::cout<<"Hello Threads!"<<std::endl;
  while(running_)
  {

    boost::system::error_code error;
    socket_.receive_from(boost::asio::buffer(boost::asio::buffer(rcv_buffer_.get(), 1289)), remote_endpoint_, 0, error);
    printf("Got some data. Steering: %f. Throttle: %f. Brake: %f", rcv_buffer_->m_steer, rcv_buffer_->m_throttle, rcv_buffer_->m_brake);
  }
}
void F1DataGrabManager::start()
{
  run_thread_ = std::thread(std::bind(&F1DataGrabManager::run_,this));
}
void F1DataGrabManager::stop()
{
  running_=false;
}
} /* namespace deepf1 */
