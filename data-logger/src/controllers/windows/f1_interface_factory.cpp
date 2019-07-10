
#include "f1_datalogger/controllers/f1_interface_factory.h"
#include "f1_datalogger/controllers/vjoy_interface.h"

std::shared_ptr<deepf1::F1Interface> deepf1::F1InterfaceFactory::getDefaultInterface(unsigned int device_id)
{
  std::shared_ptr<deepf1::VJoyInterface> rtn(new deepf1::VJoyInterface(device_id));
  return rtn;
}
