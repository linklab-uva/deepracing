#include "f1_datalogger/controllers/vigem_interface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>
#include <thread>
#include <windows.h>
#include <Xinput.h>
#include <ViGEm/Client.h>
#include <format>
namespace deepf1
{
	
VigemInterface::VigemInterface(const unsigned int& device_type, _VIGEM_CLIENT_T* client_ptr) 
{	
	if (device_type==VIGEM_DEVICE_TYPE::Xbox360){
		vigem_target_ = vigem_target_x360_alloc();
	}
	else if (device_type==VIGEM_DEVICE_TYPE::DualShock4){
		vigem_target_ = vigem_target_ds4_alloc();
	}
	else{
		std::string err_msg = std::format("Invalid device type: {}. Valid options are {} (Xbox360 controller) or {} (DualShock4 controller))", 
			device_type, (unsigned int)VIGEM_DEVICE_TYPE::Xbox360, (unsigned int)VIGEM_DEVICE_TYPE::DualShock4);
		throw std::runtime_error(err_msg);
	}
	id_ = 0;
	vigem_client_ = client_ptr;
}
VigemInterface::~VigemInterface()
{

}
void VigemInterface::pushDRS()
{

}
void VigemInterface::setCommands(const F1ControlCommand& command)
{

}
void VigemInterface::setStateDirectly(_XINPUT_STATE* gamepad_state)
{
	vigem_target_x360_update(vigem_client_, vigem_target_, *reinterpret_cast<XUSB_REPORT*>(&(gamepad_state->Gamepad)));
}
} // namespace deepf1