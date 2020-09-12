/*
 * f1_datagrab_handler.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */
#ifndef INCLUDE_UDP_LOGGING_F1_2020_DATAGRAB_HANDLER_H_
#define INCLUDE_UDP_LOGGING_F1_2020_DATAGRAB_HANDLER_H_
#include "f1_datalogger/car_data/f1_2020/timestamped_car_data.h"
#include <string>
#include <f1_datalogger/visibility_control.h>


namespace deepf1
{

class F1_DATALOGGER_UDP_LOGGING_PUBLIC IF12020DataGrabHandler
{
public:
  IF12020DataGrabHandler() = default;
  virtual ~IF12020DataGrabHandler() = default;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketCarSetupData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketCarStatusData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketCarTelemetryData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketEventData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketLapData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketMotionData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketParticipantsData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketSessionData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketFinalClassificationData& data) = 0;
  virtual void handleData(const deepf1::twenty_twenty::TimestampedPacketLobbyInfoData& data) = 0;
  virtual bool isReady() = 0;
  virtual void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) = 0;

};

} /* namespace deepf1 */

#endif /* INCLUDE_UDP_LOGGING_F1_DATAGRAB_HANDLER_H_ */
