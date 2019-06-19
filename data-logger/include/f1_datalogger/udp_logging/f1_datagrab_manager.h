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
#include "f1_datalogger/udp_logging/f1_2018_datagrab_handler.h"
#include "f1_datalogger/udp_logging/f1_protocol_versions.h"
#include <chrono>
namespace deepf1
{
class F1DataGrabManager
{
  friend class F1DataLogger;
public:
  F1DataGrabManager(std::shared_ptr<std::chrono::high_resolution_clock> clock, const std::string host = "127.0.0.1", const unsigned int port = 20777);
  virtual ~F1DataGrabManager();
private:
  void run2017(std::shared_ptr<IF1DatagrabHandler> data_handler);
  void run2018(std::shared_ptr<IF12018DataGrabHandler> data_handler);
  void start(std::shared_ptr<IF1DatagrabHandler> data_handler);
  void start(std::shared_ptr<IF12018DataGrabHandler> data_handler);
  void stop();

  static const unsigned int BUFFER_SIZE = sizeof(deepf1::twenty_eighteen::PacketMotionData);
  boost::asio::io_service io_service_;
  boost::asio::ip::udp::socket socket_;
  boost::asio::ip::udp::endpoint remote_endpoint_;
  std::thread run_thread_;
  bool running_;


  std::shared_ptr<std::chrono::high_resolution_clock> clock_;
};

} /* namespace deepf1 */

#endif /* INCLUDE_UDP_LOGGING_F1_DATAGRAB_MANAGER_H_ */
