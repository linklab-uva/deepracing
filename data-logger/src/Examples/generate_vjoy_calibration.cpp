#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <vJoy++/vjoy.h>

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
	std::string specified_direction;
	po::options_description desc("Allowed Options");

	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("direction,d", po::value<std::string>(&specified_direction)->required(), "Direction to turn the wheel, must be one of: + (steer left), - (steer right), or t (throttle), or b (brake)")
			("device_id,i", po::value<unsigned int>(&DevID)->default_value(1), "vJoy device ID to attach to")
			("sleep_time,s", po::value<unsigned int>(&sleeptime)->default_value(50), "Number of milliseconds to sleep between wheel updates")
			;
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
		if (vm.find("help") != vm.end()) {
			exit_with_help(desc);
		}
	}
	catch (const boost::exception& e) {
		exit_with_help(desc);
	}
	int direction;
	if (specified_direction.compare("+")==0)
	{
		direction = 1;
	}
	else if (specified_direction.compare("-") == 0)
	{

		direction = -1;
	}
	else if (specified_direction.compare("t") == 0 || specified_direction.compare("b") == 0)
	{
		direction = 0;
	}
	else
	{
		std::cerr << "Invalid direction specifier " << argv[2] << std::endl;
		exit(-1);
	}
	if (!vjoy_plusplus::vJoy::enabled())
	{
		std::cerr << "VJoy driver is not installed/enabled." << std::endl;
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
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (min + max)/2;
	joystick_value.wAxisY = middle;
	joystick_value.wAxisZ = 0;
	joystick_value.wAxisZRot = 0;
	vjoy->update(joystick_value);
	printf("Middle val %lu \n", middle);
	countdown(3, "Beginning calibration in");
	unsigned int da = 25;
	unsigned int dt = 50;
	if (direction > 0)
	{
		for (int angle = middle; angle >= 0; angle -= da)
		{

			printf("Setting wheel val: %ld \n", angle);
			joystick_value.wAxisY = angle;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		for(unsigned int angle = 0; angle <= middle; angle += da)
		{

			printf("Setting wheel val: %lu \n", angle);
			joystick_value.wAxisY = angle;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
	}
	else if(direction < 0)
	{
		for (unsigned int angle = middle; angle <= max; angle += da)
		{
			printf("Setting wheel val: %lu \n", angle);
			joystick_value.wAxisY = angle;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		for(unsigned int angle = max; angle >= middle; angle -= da)
		{
			printf("Setting wheel val: %lu \n", angle);
			joystick_value.wAxisY = angle;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
	}
	else if(specified_direction.compare("t") == 0)
	{
		for (unsigned int throttleval = 0; throttleval <= max; throttleval += 50)
		{
			printf("Setting throttle val: %lu \n", throttleval);
			joystick_value.wAxisZ = throttleval;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
		for (int throttleval = max; throttleval >= 0; throttleval -= 50)
		{
			printf("Setting throttle val: %ld \n", throttleval);
			joystick_value.wAxisZ = throttleval;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
	}
	else if (specified_direction.compare("b") == 0)
	{
		for (unsigned int brakeval = 0; brakeval <= max; brakeval += 50)
		{
			printf("Setting brake val: %lu \n", brakeval);
			joystick_value.wAxisZRot = brakeval;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
		for (int brakeval = max; brakeval >= 0; brakeval -= 50)
		{
			printf("Setting brake val: %ld \n", brakeval);
			joystick_value.wAxisZRot = brakeval;
			vjoy->update(joystick_value);
			std::this_thread::sleep_for(std::chrono::microseconds(sleeptime));
		}
	}
	joystick_value.wAxisY = middle;
	joystick_value.wAxisZ = 0;
	joystick_value.wAxisZRot = 0;
	vjoy->update(joystick_value);
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

}