/*
 * f1_datalogger.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_F1_DATALOGGER_H_ 
#define INCLUDE_F1_DATALOGGER_H_
#include <f1_datalogger/visibility_control.h>
#ifdef F1_DATALOGGER_PROTO_DLL_MACRO
#endif
#include "image_logging/f1_framegrab_manager.h"
#include "udp_logging/f1_datagrab_manager.h"
namespace deepf1
{

class F1_DATALOGGER_PUBLIC F1DataLogger
{
public:
  F1DataLogger(const std::string& search_string, std::string host="127.0.0.1", unsigned int port= 20777);
  virtual ~F1DataLogger();

  static void countdown(unsigned int seconds, std::string txt="");
  void start(double capture_frequency, std::shared_ptr<IF1FrameGrabHandler> image_handler);
  void add2018UDPHandler(std::shared_ptr<IF12018DataGrabHandler> udp_handler);
  void add2020UDPHandler(std::shared_ptr<IF12020DataGrabHandler> udp_handler);
  void stop();

  const deepf1::TimePoint getStart() const;
private:
  std::shared_ptr<F1FrameGrabManager> frame_grab_manager_;
  std::shared_ptr<F1DataGrabManager> data_grab_manager_;


  deepf1::TimePoint begin_;

  std::string host_;
  unsigned int port_;
  
  ClockPtr clock_;

};

} /* namespace deepf1 */

#endif /* INCLUDE_F1_DATALOGGER_H_ */
