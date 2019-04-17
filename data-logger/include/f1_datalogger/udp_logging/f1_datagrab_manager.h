/*
 * f1_datagrab_manager.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_UDP_LOGGING_F1_DATAGRAB_MANAGER_H_
#define INCLUDE_UDP_LOGGING_F1_DATAGRAB_MANAGER_H_
#include <boost/asio.hpp>
#include <thread>
#include <memory>
#include "f1_datalogger/udp_logging/f1_datagrab_handler.h"
#include <chrono>
namespace deepf1
{

class F1DataGrabManager
{
public:
  F1DataGrabManager(std::shared_ptr<std::chrono::high_resolution_clock> clock,
                    std::shared_ptr<IF1DatagrabHandler> handler, const std::string host = "127.0.0.1",
                    const unsigned int port = 20777);
  virtual ~F1DataGrabManager();

  void start();
  void stop();
  static const unsigned int BUFFER_SIZE = sizeof(deepf1::UDPPacket);	
private:
  boost::asio::io_service io_service_;
  boost::asio::ip::udp::socket socket_;
  boost::asio::ip::udp::endpoint remote_endpoint_;
  std::thread run_thread_;
  std::shared_ptr<IF1DatagrabHandler> data_handler_;
  bool running_;
  static const unsigned int sleeptime = 2;


  std::shared_ptr<std::chrono::high_resolution_clock> clock_;

  void run_();
};

} /* namespace deepf1 */

#endif /* INCLUDE_UDP_LOGGING_F1_DATAGRAB_MANAGER_H_ */
