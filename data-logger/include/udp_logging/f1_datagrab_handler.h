/*
 * f1_datagrab_handler.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_UDP_LOGGING_F1_DATAGRAB_HANDLER_H_
#define INCLUDE_UDP_LOGGING_F1_DATAGRAB_HANDLER_H_
#include "car_data/timestamped_car_data.h"
namespace deepf1
{

class IF1DatagrabHandler
{
public:
  IF1DatagrabHandler() = default;
  virtual ~IF1DatagrabHandler() = default;
  virtual void handleData(const deepf1::TimestampedUDPData& data) = 0;
  virtual bool isReady() = 0;
  virtual void init(const std::chrono::high_resolution_clock::time_point& begin) = 0;

};

} /* namespace deepf1 */

#endif /* INCLUDE_UDP_LOGGING_F1_DATAGRAB_HANDLER_H_ */
