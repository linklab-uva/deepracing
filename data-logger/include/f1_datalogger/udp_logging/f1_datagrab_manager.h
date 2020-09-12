/*
 * f1_datagrab_manager.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_UDP_LOGGING_F1_DATAGRAB_MANAGER_H_
#define INCLUDE_UDP_LOGGING_F1_DATAGRAB_MANAGER_H_
#include <f1_datalogger/udp_logging/visibility_control.h>
#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <thread>
#include <memory>
#include "f1_datalogger/udp_logging/f1_datagrab_handler.h"
#include "f1_datalogger/udp_logging/f1_2018_datagrab_handler.h"
#include "f1_datalogger/udp_logging/f1_2020_datagrab_handler.h"
#include "f1_datalogger/udp_logging/f1_protocol_versions.h"
#include <chrono>
#include <map>

namespace deepf1
{
class F1_DATALOGGER_UDP_LOGGING_PUBLIC F1DataGrabManager
{
  friend class F1DataLogger;
public:
  F1DataGrabManager(const deepf1::TimePoint& begin, const std::string host = "127.0.0.1", const unsigned int port = 20777);
  virtual ~F1DataGrabManager();
   const std::map<deepf1::uint8, std::string> packetIdMap = { {0,"MOTION"}, {1,"SESSION"}, {2,"LAPDATA"}, {3,"EVENT"}, {4,"PARTICIPANTS"}, {5,"CARSETUPS"}, {6,"CARTELEMETRY"}, {7,"CARSTATUS"}, {8,"FINAL_CLASSIFICATION"}, {9,"LOBBY_INFO"} };
private:
  void run();
  void start();
  void stop();

  static constexpr unsigned int BUFFER_SIZE = sizeof(deepf1::twenty_eighteen::PacketMotionData);
  boost::asio::io_service io_service_;
  boost::asio::ip::udp::socket socket_;
  boost::asio::ip::udp::endpoint remote_endpoint_;
  std::thread run_thread_;
  bool running_;

  deepf1::TimePoint begin_;
  std::vector< std::shared_ptr<IF12018DataGrabHandler> > handlers2018;
  std::vector< std::shared_ptr<IF12020DataGrabHandler> > handlers2020;
};

} /* namespace deepf1 */

#endif /* INCLUDE_UDP_LOGGING_F1_DATAGRAB_MANAGER_H_ */
