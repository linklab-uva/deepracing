#include "f1_datalogger/controllers/vigem_interface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>
#include <thread>
#include <windows.h>
#include <ViGEm/Client.h>
#include <format>

deepf1::VigemInterface::VigemInterface(const unsigned int& device_type) 
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
}
deepf1::VigemInterface::~VigemInterface()
{

}
void deepf1::VigemInterface::pushDRS()
{

}
void deepf1::VigemInterface::setCommands(const F1ControlCommand& command)
{

}
