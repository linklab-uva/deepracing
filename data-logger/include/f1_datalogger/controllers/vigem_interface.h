#ifndef INCLUDE_CONTROLLERS_VJOY_INTERFACE_H_
#define INCLUDE_CONTROLLERS_VJOY_INTERFACE_H_
#include <cmath>
#include <f1_datalogger/controllers/f1_interface.h>
#include <stdint.h>
#include <f1_datalogger/controllers/internal/vigem_decl.h>
#include <mutex>

namespace deepf1 {

	class MultiagentF1InterfaceFactory;
	class F1_DATALOGGER_CONTROLS_PUBLIC VigemInterface : public F1Interface
	{
	friend class MultiagentF1InterfaceFactory;
	public:
	  virtual ~VigemInterface();
	  void setCommands(const F1ControlCommand& command) override;
	  void pushDRS() override;

	  void setStateDirectly(XINPUT_STATE& gamepad_state);


	private:
	  VigemInterface(const unsigned int& device_type, _VIGEM_CLIENT_T* client_ptr, uint64_t id);
	  PVIGEM_TARGET vigem_target_;
	  PVIGEM_CLIENT vigem_client_;
	  uint8_t device_type_;
	  uint64_t id_;
	  std::mutex update_mutex_;
	  XINPUT_STATE current_controller_state_;
	};
	
}
#endif
