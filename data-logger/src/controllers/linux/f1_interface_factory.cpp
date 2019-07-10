
#include "f1_datalogger/controllers/f1_interface_factory.h"
#include <stdexcept>
std::shared_ptr<deepf1::F1Interface> deepf1::F1InterfaceFactory::getDefaultInterface()
{
  throw std::runtime_error("Virtual Game interface is not supported on Linux in this version... :( Sorry...");
}
