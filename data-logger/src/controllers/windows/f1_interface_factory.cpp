
#include "f1_datalogger/controllers/f1_interface_factory.h"
#include "f1_datalogger/controllers/vjoy_interface.h"

std::shared_ptr<deepf1::F1Interface> deepf1::F1InterfaceFactory::getDefaultInterface()
{
  std::shared_ptr<deepf1::VJoyInterface> rtn(new deepf1::VJoyInterface);
  return rtn;
}
