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
	if (argc > 1)
	{
		search = std::string(argv[1]);
	}
	double sleeptime = 0.5;
	if (argc > 2)
	{
		sleeptime = atof(argv[2]);
	}
	if (argc > 3)
	{
		outfile = std::string(argv[3]);
	}
	unsigned long milliseconds = (unsigned long)std::round(sleeptime*1000.0);
	std::shared_ptr<VJoyCalibration_FrameGrabHandler> image_handler(new VJoyCalibration_FrameGrabHandler());
	std::shared_ptr<VJoyCalibration_DataGrabHandler> udp_handler(new VJoyCalibration_DataGrabHandler());
	deepf1::F1DataLogger dl(search, image_handler, udp_handler);
	dl.start();
	std::unique_ptr<vjoy_plusplus::vJoy> vjoy(new vjoy_plusplus::vJoy(1));
	vjoy_plusplus::JoystickPosition iReport;
	iReport.lButtons = 0x00000000;
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (min + max) / 2;
	iReport.wAxisY = 0;
	iReport.wAxisZ = 0;
	iReport.wAxisZRot = 0;
	vjoy->update(iReport);
	std::this_thread::sleep_for(std::chrono::seconds(3));
	std::ofstream ostream(outfile);
	for(unsigned long angle = 0; angle <=max; angle+=25)
	{
		iReport.wAxisY = angle;
		vjoy->update(iReport);
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
		deepf1::UDPPacket current_packet_ = udp_handler->getCurrentPacket();
		printf("Current vJoy value: %ld\n", angle);
		printf("Current Data: Steering: %f. Throttle: %f. Brake: %f. Lap Time: %f\n", current_packet_.m_steer, current_packet_.m_throttle, current_packet_.m_brake, current_packet_.m_lapTime);
		ostream << angle << "," << current_packet_.m_steer << std::endl;

	}
	ostream.close();
	//cv::waitKey(0);

}

