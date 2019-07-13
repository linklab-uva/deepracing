
#include "f1_datalogger/controllers/f1_interface_factory.h"
#include <stdexcept>
std::shared_ptr<deepf1::F1Interface> deepf1::F1InterfaceFactory::getDefaultInterface(unsigned int device_id)
{
  throw std::runtime_error("Virtual game interface is not supported on IOS in this version... :( Sorry...");
}
