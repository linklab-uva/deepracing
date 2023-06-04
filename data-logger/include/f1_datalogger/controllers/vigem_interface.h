#ifndef INCLUDE_CONTROLLERS_VJOY_INTERFACE_H_
#define INCLUDE_CONTROLLERS_VJOY_INTERFACE_H_
#include <cmath>
#include <f1_datalogger/controllers/f1_interface.h>
#include <stdint.h>
#include <f1_datalogger/controllers/internal/vigem_decl.h>

struct _XINPUT_STATE;
namespace deepf1 {

	class MultiagentF1InterfaceFactory;
	class F1_DATALOGGER_CONTROLS_PUBLIC VigemInterface : public F1Interface
	{
	friend class MultiagentF1InterfaceFactory;
	public:
	  virtual ~VigemInterface();
	  void setCommands(const F1ControlCommand& command) override;
	  void pushDRS() override;

	  void setStateDirectly(_XINPUT_STATE& gamepad_state);


	private:
	  VigemInterface(const unsigned int& device_type, _VIGEM_CLIENT_T* client_ptr);
	  _VIGEM_TARGET_T* vigem_target_;
	  _VIGEM_CLIENT_T* vigem_client_;
	  uint64_t id_;
	};
	
}
#endif
