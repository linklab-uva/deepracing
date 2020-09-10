/*
 * f1_datagrab_handler.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_UDP_LOGGING_F1_DATAGRAB_HANDLER_H_
#define INCLUDE_UDP_LOGGING_F1_DATAGRAB_HANDLER_H_
#include "f1_datalogger/car_data/f1_2018/timestamped_car_data.h"
#include <string>
#include <f1_datalogger/visibility_control.h>

namespace deepf1
{

class F1_DATALOGGER_PUBLIC IF1DatagrabHandler
{
public:
  IF1DatagrabHandler() = default;
  virtual ~IF1DatagrabHandler() = default;
  virtual void handleData(const deepf1::TimestampedUDPData& data) = 0;
  virtual bool isReady() = 0;
  virtual void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) = 0;

};

} /* namespace deepf1 */

#endif /* INCLUDE_UDP_LOGGING_F1_DATAGRAB_HANDLER_H_ */
