/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/f1_datalogger.h"
 //#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vJoy++/vjoy.h>

void countdown(unsigned int seconds, std::string text = "")
{
	std::cout << text << std::endl;
	for (unsigned int i = seconds; i > 0; i--)
	{
		std::cout << i << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}
class VJoyCalibration_DataGrabHandler : public deepf1::IF1DatagrabHandler
{
public:
	VJoyCalibration_DataGrabHandler()
	{

	}
	bool isReady() override
	{
		return true;
	}
	void handleData(const deepf1::TimestampedUDPData& data) override
	{
		current_packet_ = deepf1::UDPPacket(data.data);
	}
	void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override
	{
		this->begin = begin;
	}
	deepf1::UDPPacket getCurrentPacket()
	{
		return current_packet_;
	}
private:
	std::chrono::high_resolution_clock::time_point begin;
	deepf1::UDPPacket current_packet_;
};
class VJoyCalibration_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
	VJoyCalibration_FrameGrabHandler()
	{
		
	}
	virtual ~VJoyCalibration_FrameGrabHandler()
	{
		
	}
	bool isReady() override
	{
		return true;
	}
	void handleData(const deepf1::TimestampedImageData& data) override
	{

	}
	void init(const std::chrono::high_resolution_clock::time_point& begin, const cv::Size& window_size) override
	{
		
	}
private:
};
int main(int argc, char** argv)
{
	std::string search = "2017";
	std::string outfile = "out.csv";
	double sleeptime = 0.5;
	if (argc > 1)
	{
		sleeptime = atof(argv[1]);
	}
	if (argc > 2)
	{
		outfile = std::string(argv[2]);
	}
	unsigned long milliseconds = (unsigned long)std::round(sleeptime*1000.0);
	std::shared_ptr<VJoyCalibration_FrameGrabHandler> image_handler(new VJoyCalibration_FrameGrabHandler());
	std::shared_ptr<VJoyCalibration_DataGrabHandler> udp_handler(new VJoyCalibration_DataGrabHandler());
	deepf1::F1DataLogger dl(search, image_handler, udp_handler);
	dl.start();
	std::unique_ptr<vjoy_plusplus::vJoy> vjoy(new vjoy_plusplus::vJoy(1));
	vjoy_plusplus::JoystickPosition joystick_value;
	joystick_value.lButtons = 0x00000000;
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (min + max) / 2;
	joystick_value.wAxisX = 0;
	joystick_value.wAxisY = 0;
	joystick_value.wAxisXRot = 0;
	joystick_value.wAxisYRot = 0;
	vjoy->update(joystick_value);
	countdown(3, "Testing calibration in");
	std::ofstream ostream(outfile);
	//Best fit line is : y = -16383.813867*x + 16383.630437
	for(int vjoyangle = max; vjoyangle >= 0; vjoyangle -=25)
	{
		joystick_value.wAxisX = vjoyangle;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
		deepf1::UDPPacket current_packet_ = udp_handler->getCurrentPacket();
		printf("Input Angle: %d\n", vjoyangle);
		printf("Current Steering: %f\n.", current_packet_.m_steer);
		ostream << current_packet_.m_steer << "," << vjoyangle << std::endl;
	}
	joystick_value.wAxisX = 0;
	joystick_value.wAxisY = 0;
	joystick_value.wAxisXRot = 0;
	joystick_value.wAxisYRot = 0;
	vjoy->update(joystick_value);
	for (int vjoyangle = 0; vjoyangle <= max; vjoyangle += 25)
	{
		joystick_value.wAxisY = vjoyangle;
		vjoy->update(joystick_value);
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
		deepf1::UDPPacket current_packet_ = udp_handler->getCurrentPacket();
		printf("Input Angle: %d\n", vjoyangle);
		printf("Current Steering: %f\n.", current_packet_.m_steer);
		ostream << current_packet_.m_steer << "," << vjoyangle << std::endl;
	}
	ostream.close();
	//cv::waitKey(0);

}

