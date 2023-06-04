#include "f1_datalogger/controllers/multiagent_f1_interface_factory.h"
#include <windows.h>
#include <ViGEm/Client.h>
#include <sstream>
#include <f1_datalogger/controllers/vigem_interface.h>
#include <format>

namespace deepf1
{
  MultiagentF1InterfaceFactory::MultiagentF1InterfaceFactory()
  {
    vigem_client_ = vigem_alloc();
    const VIGEM_ERROR return_code = vigem_connect(vigem_client_);
    if (!VIGEM_SUCCESS(return_code))
    {
      vigem_free(vigem_client_);
      std::string err_msg = std::format("Unable to connect to driver. Error code: {:#08x}", (uint32_t)return_code);
      throw std::runtime_error(err_msg);
    }
    uuid_ = 0;
  }
  MultiagentF1InterfaceFactory::~MultiagentF1InterfaceFactory()
  {
    for(std::pair<uint64_t, std::shared_ptr<VigemInterface>> pair : created_interfaces_)
    {
      std::shared_ptr<VigemInterface> interface = pair.second;
      if (!(interface->vigem_target_==nullptr))
      {
        vigem_target_remove(vigem_client_, interface->vigem_target_);
        vigem_target_free(interface->vigem_target_);
      }
    }
    if (!(vigem_client_==nullptr))
    {
      vigem_disconnect(vigem_client_);
      vigem_free(vigem_client_);
    }
  }
  bool MultiagentF1InterfaceFactory::disconnectInterface(std::shared_ptr<F1Interface> interface)
  {
    if(!interface)
    {
      return false;
    }
    std::shared_ptr<VigemInterface> downcasted = std::static_pointer_cast<VigemInterface>(interface);
    if(!downcasted)
    {
      return false;
    }
    if (!(downcasted->vigem_target_==nullptr))
    {
      vigem_target_remove(vigem_client_, downcasted->vigem_target_);
      vigem_target_free(downcasted->vigem_target_);
      created_interfaces_.erase(downcasted->id_);
      return true;
    }
    return false;
  }
  std::shared_ptr<F1Interface> MultiagentF1InterfaceFactory::createInterface(unsigned int device_id)
  {
    std::shared_ptr<VigemInterface> rtn(new VigemInterface(device_id, vigem_client_, uuid_++));
    const VIGEM_ERROR return_code = vigem_target_add(vigem_client_, rtn->vigem_target_);
    if (!VIGEM_SUCCESS(return_code))
    {
      vigem_target_free(rtn->vigem_target_);
      std::string err_msg = std::format("Target plugin failed with error code: {:#08x}", (uint32_t)return_code);
      throw std::runtime_error(err_msg);
    }
    created_interfaces_[rtn->id_]=rtn;
    return rtn;
  }
}