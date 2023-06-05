#include "f1_datalogger/controllers/vigem_interface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>
#include <thread>
#include <format>
namespace deepf1
{
	
VigemInterface::VigemInterface(const unsigned int& device_type, PVIGEM_CLIENT client_ptr, uint64_t id) 
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
	current_controller_state_.dwPacketNumber=0;
	current_controller_state_.Gamepad.bLeftTrigger=current_controller_state_.Gamepad.bLeftTrigger=0;
	current_controller_state_.Gamepad.sThumbLX=current_controller_state_.Gamepad.sThumbLY=
		current_controller_state_.Gamepad.sThumbRX=current_controller_state_.Gamepad.sThumbRY=0;
	current_controller_state_.Gamepad.wButtons=0;
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
void VigemInterface::setStateDirectly(XINPUT_STATE& gamepad_state)
{
	switch (device_type_)
	{
	case VIGEM_DEVICE_TYPE::Xbox360:
	{
		std::scoped_lock<std::mutex> lock(update_mutex_);
		current_controller_state_ = gamepad_state;
		vigem_target_x360_update(vigem_client_, vigem_target_, *reinterpret_cast<XUSB_REPORT*>(&(current_controller_state_.Gamepad)));	
	}
		break;
	default:
		break;
	}
}
} // namespace deepf1