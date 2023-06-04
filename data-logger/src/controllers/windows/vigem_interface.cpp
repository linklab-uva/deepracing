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
	
VigemInterface::VigemInterface(const unsigned int& device_type, _VIGEM_CLIENT_T* client_ptr, uint64_t id) 
{	
	if (device_type==VIGEM_DEVICE_TYPE::Xbox360){
		vigem_target_ = vigem_target_x360_alloc();
		device_type_ = VIGEM_DEVICE_TYPE::Xbox360;
	}
	else{
		std::string err_msg = std::format("Invalid device type: {}. Currently, only Xbox360({}) is supported. DualShock4 support is in-work :)", 
			device_type, (unsigned int)VIGEM_DEVICE_TYPE::Xbox360);
		throw std::runtime_error(err_msg);
	}
	id_ = id;
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
void VigemInterface::setStateDirectly(_XINPUT_STATE& gamepad_state)
{
	switch (device_type_)
	{
	case VIGEM_DEVICE_TYPE::Xbox360:
		vigem_target_x360_update(vigem_client_, vigem_target_, *reinterpret_cast<XUSB_REPORT*>(&(gamepad_state.Gamepad)));
		break;
	default:
		break;
	}
}
} // namespace deepf1