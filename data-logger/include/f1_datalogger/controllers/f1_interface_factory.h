#ifndef INCLUDE_CONTROLLERS_F1_INTERFACE_FACTORY_H_
#define INCLUDE_CONTROLLERS_F1_INTERFACE_FACTORY_H_
#include "f1_datalogger/controllers/f1_interface.h"
#include <memory>
namespace deepf1
{
class F1InterfaceFactory
{

public:
  static std::shared_ptr<F1Interface> getDefaultInterface();
  
};
}

#endif
