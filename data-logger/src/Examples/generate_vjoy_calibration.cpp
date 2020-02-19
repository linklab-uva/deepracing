#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <vJoy_plusplus/vjoy.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
void exit_with_help(po::options_description& desc)
{
	std::stringstream ss;
	ss << "vJoy F1 Calibration. Command line arguments are as follows:" << std::endl;
	desc.print(ss);
	std::printf("%s", ss.str().c_str());
	exit(0); 
}
void countdown(unsigned int seconds, std::string text="")
{
	std::cout << text << std::endl;
	for (unsigned int i = seconds; i > 0; i--)
	{
		std::cout << i << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}
int main(int argc, char** argv)
{
	std::cout<<"Hello World!"<<std::endl; 
	unsigned int DevID, sleeptime;
	std::string specified_control;
	po::options_description desc("Allowed Options");

	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("control,c", po::value<std::string>(&specified_control)->required(), "Control input to calibrate, must be one of: + (steer left), - (steer right), or t (throttle), or b (brake)")
			("device_id,d", po::value<unsigned int>(&DevID)->default_value(1), "vJoy device ID to attach to")
			("sleep_time,s", po::value<unsigned int>(&sleeptime)->default_value(50), "Number of milliseconds to sleep between wheel updates")
			;
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
		if (vm.find("help") != vm.end()) 
		{
			exit_with_help(desc);
		}
	}
	catch (const boost::exception& e) {
		exit_with_help(desc);
	}
	int direction;
	if (specified_control.compare("+")==0)
	{
		direction = 1;
	}
	else if (specified_control.compare("-") == 0)
	{

		direction = -1;
	}
	else if (specified_control.compare("t") == 0 || specified_control.compare("b") == 0)
	{
		direction = 0;
	}
	else if (specified_control.compare("p") == 0)
	{
		direction = 0;
	}
	else if (specified_control.compare("c") == 0)
	{
		direction = 0;
	}
	else if (specified_control.compare("gu") == 0)
	{
		direction = 0;
	}
	else if (specified_control.compare("gd") == 0)
	{
		direction = 0;
	}
	else if (specified_control.compare("drs") == 0)
	{
		direction = 0;
	}
	else
	{
		std::cerr << "Invalid direction specifier " << specified_control << std::endl;
		exit(-1);
	}
	if (!vjoy_plusplus::vJoy::enabled())
	{
		std::cerr << "vJoy driver is not installed/enabled." << std::endl;
		exit(-1);
	}
	vjoy_plusplus::VjoyDeviceStatus status = vjoy_plusplus::vJoy::getStatus(DevID);
	switch (status)
	{
		case vjoy_plusplus::VjoyDeviceStatus::VJD_STAT_OWN:
			std::printf("vJoy device %d is already owned by this feeder\n", DevID);
			break;
		case vjoy_plusplus::VjoyDeviceStatus::VJD_STAT_FREE:
			std::printf("vJoy device %d is free\n", DevID);
			break;
		case vjoy_plusplus::VjoyDeviceStatus::VJD_STAT_BUSY:
			std::printf("vJoy device %d is already owned by another feeder\nCannot continue\n", DevID);
			exit(-3);
		case vjoy_plusplus::VjoyDeviceStatus::VJD_STAT_MISS:
			std::printf("vJoy device %d is not installed or disabled\nCannot continue\n", DevID);
			exit(-4);
		default:
			std::printf("vJoy device %d general error\nCannot continue\n", DevID);
			exit(-5);
	}

	vjoy_plusplus::JoystickPosition joystick_value;
	std::unique_ptr<vjoy_plusplus::vJoy> vjoy(new vjoy_plusplus::vJoy(DevID));

	joystick_value.lButtons = 0x00000000;
	unsigned int da = 12;
	unsigned int dt = 50;
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	countdown(3, "Beginning calibration in");
	joystick_value.wAxisX = 0;
	joystick_value.wAxisY = 0;
	joystick_value.wAxisXRot = 0;
	joystick_value.wAxisYRot = 0;
	vjoy->update(joystick_value);
	if (direction > 0)
	{
		for (unsigned int angle = 0; angle <= max; angle += da)
		{

			printf("Setting wheel val: %ld \n", angle);
			joystick_value.wAxisX = angle;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
		joystick_value.wAxisX = max;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		for(unsigned int angle = max; angle > da; angle -= da)
		{

			printf("Setting wheel val: %lu \n", angle);
			joystick_value.wAxisX = angle;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
	}
	else if(direction < 0)
	{
		for (unsigned int angle = 0; angle < max; angle += da)
		{
			printf("Setting wheel val: %lu \n", angle);
			joystick_value.wAxisY = angle;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
		joystick_value.wAxisY = max;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		for (unsigned int angle = max; angle > da; angle -= da)
		{

			printf("Setting wheel val: %lu \n", angle);
			joystick_value.wAxisY = angle;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
	}
	else if(specified_control.compare("t") == 0)
	{
		for (unsigned int throttleval = 0; throttleval < max; throttleval += dt)
		{
			printf("Setting throttle val: %lu \n", throttleval);
			joystick_value.wAxisXRot = throttleval;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
		joystick_value.wAxisXRot = max;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		for (int throttleval = max; throttleval >= 0; throttleval -= dt)
		{
			printf("Setting throttle val: %ld \n", throttleval);
			joystick_value.wAxisXRot = throttleval;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
	}
	else if (specified_control.compare("b") == 0)
	{
		for (unsigned int brakeval = 0; brakeval < max; brakeval += dt)
		{
			printf("Setting brake val: %lu \n", brakeval);
			joystick_value.wAxisYRot = brakeval;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
		joystick_value.wAxisYRot = max;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		for (int brakeval = max; brakeval >= 0; brakeval -= dt)
		{
			printf("Setting brake val: %ld \n", brakeval);
			joystick_value.wAxisYRot = brakeval;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
	}
	else if (specified_control.compare("p") == 0)
	{
		joystick_value.lButtons = 0x1;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		joystick_value.lButtons = 0x0;
		vjoy->update(joystick_value);
	}
	else if (specified_control.compare("c") == 0)
	{
		joystick_value.lButtons = 0x1<<1;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		joystick_value.lButtons = 0x0;
		vjoy->update(joystick_value);
	}
	else if (specified_control.compare("gu") == 0)
	{
		joystick_value.lButtons =  0x1<<2;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		joystick_value.lButtons = 0x0;
		vjoy->update(joystick_value);
	}
	else if (specified_control.compare("gd") == 0)
	{
		joystick_value.lButtons = 0x1<<3;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		joystick_value.lButtons = 0x0;
		vjoy->update(joystick_value);
	}
	else if (specified_control.compare("drs") == 0)
	{
		joystick_value.lButtons = 0x1<<4;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(125));
		joystick_value.lButtons = 0x0;
		vjoy->update(joystick_value);
	}
	std::printf("Reset all values to 0\n");
	joystick_value.wAxisX = 0;
	joystick_value.wAxisY = 0;
	joystick_value.wAxisXRot = 0;
	joystick_value.wAxisYRot = 0;
	vjoy->update(joystick_value);
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

}