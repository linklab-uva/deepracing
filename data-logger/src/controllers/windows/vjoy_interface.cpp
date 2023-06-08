#include "f1_datalogger/controllers/vjoy_interface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>
#include <thread>


deepf1::VJoyInterface::VJoyInterface(const unsigned int& device_id) : vjoy_(device_id)
{
	max_vjoysteer_ = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	max_vjoythrottle_ = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	max_vjoybrake_ = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	current_js.lButtons = 0x00000000;
	std::cout << "Vjoy Enabled: " << vjoy_.enabled() << std::endl;
}


deepf1::VJoyInterface::~VJoyInterface()
{
}
void deepf1::VJoyInterface::setButtons(long bitmask)
{
	std::scoped_lock lock(mutex_);
	current_js.lButtons = bitmask;
	vjoy_.update(current_js);
}
XINPUT_STATE deepf1::VJoyInterface::getCurrentState( )
{
	std::scoped_lock lock(mutex_);
	XINPUT_STATE rtn;

	return rtn;
}
void deepf1::VJoyInterface::setStateDirectly(const XINPUT_STATE& gamepad_state)
{
	
	std::scoped_lock lock(mutex_);
	double left_thumb_x = (double)gamepad_state.Gamepad.sThumbLX;
	if (left_thumb_x>=0.0)
	{
		current_js.wAxisX = 0;
		current_js.wAxisY = std::round(max_vjoysteer_*(left_thumb_x/32767.0));
	}
	else
	{
		current_js.wAxisX = std::round(max_vjoysteer_*(-left_thumb_x/32767.0));
		current_js.wAxisY = 0;
	}
	double right_trigger = (double)gamepad_state.Gamepad.bRightTrigger;
	double left_trigger = (double)gamepad_state.Gamepad.bLeftTrigger;
	current_js.wAxisXRot = std::round(max_vjoythrottle_*(right_trigger/255.0));
	//std::printf("Requested brake: %f\n", command.brake);
	current_js.wAxisYRot = std::round(max_vjoybrake_*(left_trigger/255.0));
	current_js.lButtons = gamepad_state.Gamepad.wButtons;
	vjoy_.update(current_js);
}
void deepf1::VJoyInterface::pushDRS()
{
	std::scoped_lock lock(mutex_);
	current_js.lButtons = 0x1<<4;
	vjoy_.update(current_js);
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	current_js.lButtons = 0x0;
	vjoy_.update(current_js);
}

void deepf1::VJoyInterface::setCommands(const F1ControlCommand& command)
{
	std::scoped_lock lock(mutex_);
	if (command.steering > 0)
	{
		current_js.wAxisX = std::round(max_vjoysteer_*command.steering);
		current_js.wAxisY = 0;
	}
	else if(command.steering < 0)
	{
		current_js.wAxisX = 0;
		current_js.wAxisY = std::round(max_vjoysteer_ * std::abs(command.steering));
	}
	else
	{
		current_js.wAxisX = 0;
		current_js.wAxisY = 0;
	}
	current_js.wAxisXRot = std::round(max_vjoythrottle_*command.throttle);
	//std::printf("Requested brake: %f\n", command.brake);
	current_js.wAxisYRot = std::round(max_vjoybrake_*command.brake);
	//std::printf("Setting yrot to: %f\n", js.wAxisYRot);
	vjoy_.update(current_js);
}
