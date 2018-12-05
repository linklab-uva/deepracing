/*
 * f1_datalogger.h
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#ifndef INCLUDE_F1_DATALOGGER_H_
#define INCLUDE_F1_DATALOGGER_H_
#include "image_logging/f1_framegrab_manager.h"
namespace deepf1
{

class F1DataLogger
{
public:
  F1DataLogger(std::shared_ptr<scl::Window> window);
  F1DataLogger(const std::string& search_string);
  virtual ~F1DataLogger();
private:

};

} /* namespace deepf1 */

#endif /* INCLUDE_F1_DATALOGGER_H_ */
