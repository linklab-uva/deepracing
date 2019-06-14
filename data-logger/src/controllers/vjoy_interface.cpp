#include "f1_datalogger/controllers/vjoy_interface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>
deepf1::VJoyInterface::VJoyInterface(const unsigned int& device_id) : vjoy_(device_id)
{
	max_vjoysteer_ = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	max_vjoythrottle_ = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	max_vjoybrake_ = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	std::cout << "Vjoy Enabled: " << vjoy_.enabled() << std::endl;
}


deepf1::VJoyInterface::~VJoyInterface()
{
}

void deepf1::VJoyInterface::setCommands(const F1ControlCommand& command)
{
	vjoy_plusplus::JoystickPosition js;
	js.lButtons = 0x00000000;
	if (command.steering > 0)
	{
		js.wAxisX = std::round(max_vjoysteer_*command.steering);
		js.wAxisY = 0;
	}
	else if(command.steering < 0)
	{
		js.wAxisX = 0;
		js.wAxisY = std::round(max_vjoysteer_ * std::abs(command.steering));
	}
	else
	{
		js.wAxisX = 0;
		js.wAxisY = 0;
	}
	js.wAxisXRot = std::round(max_vjoythrottle_*command.throttle);
	js.wAxisYRot = std::round(max_vjoybrake_*command.brake);
	vjoy_.update(js);
}
