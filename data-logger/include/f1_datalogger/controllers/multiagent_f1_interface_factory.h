#ifndef INCLUDE_CONTROLLERS_MULTIAGENT_F1_INTERFACE_FACTORY_H_
#define INCLUDE_CONTROLLERS_MULTIAGENT_F1_INTERFACE_FACTORY_H_
#include "f1_datalogger/controllers/f1_interface.h"
#include <memory>
#include <unordered_map>
#include <f1_datalogger/controllers/internal/vigem_decl.h>

namespace deepf1
{
  class VigemInterface;   
  class F1_DATALOGGER_CONTROLS_PUBLIC MultiagentF1InterfaceFactory
  {
  public:
    MultiagentF1InterfaceFactory();
    ~MultiagentF1InterfaceFactory();
    std::shared_ptr<F1Interface> createInterface(unsigned int device_id = 1);
    bool disconnectInterface(std::shared_ptr<F1Interface> iface);

  private:
    _VIGEM_CLIENT_T* vigem_client_;
    std::unordered_map< uint64_t, std::shared_ptr<VigemInterface> > created_interfaces_;
    uint64_t uuid_;
    
  };

}
#endif