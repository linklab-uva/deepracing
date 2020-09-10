/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/f1_datalogger.h"
#include "f1_datalogger/udp_logging/common/measurement_handler_2018.h"
#include "f1_datalogger/controllers/vjoy_interface.h"
 //#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <fstream>
#include <vJoy_plusplus/vjoy.h>

void countdown(unsigned int seconds, std::string text = "")
{
	std::cout << text << std::endl;
	for (unsigned int i = seconds; i > 0; i--)
	{
		std::cout << i << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}
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
	std::string search = "2018";
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
	std::shared_ptr<deepf1::MeasurementHandler2018> udp_handler(new deepf1::MeasurementHandler2018());
	deepf1::F1DataLogger dl(search);
	dl.add2018UDPHandler(udp_handler);
	dl.start(60.0, image_handler);
	std::unique_ptr<deepf1::VJoyInterface> vjoyInterface(new deepf1::VJoyInterface);
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (min + max) / 2;
	deepf1::F1ControlCommand commands;
	vjoyInterface->setCommands(commands);
	countdown(3, "Testing calibration in");
	std::ofstream ostream(outfile);
	//Best fit line is : y = -16383.813867*x + 16383.630437
	for(float steer = -1.0; steer <= 1.0; steer+=0.01)
	{
		commands.steering = steer;
		vjoyInterface->setCommands(commands);
		std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
		deepf1::twenty_eighteen::PacketCarTelemetryData current_telemetry_packet_ = udp_handler->getCurrentTelemetryData().data;
		deepf1::twenty_eighteen::PacketMotionData current_motion_packet_ = udp_handler->getCurrentMotionData().data;
		float vjoy_angle = commands.steering*double(max);
		float steering_ratio = current_telemetry_packet_.m_carTelemetryData[0].m_steer;
		float front_wheels_angle = current_motion_packet_.m_frontWheelsAngle;

		printf("Vjoy Input Angle: %f\n", vjoy_angle);
		printf("Steering Ratio: %f\n.", steering_ratio);
		printf("Current Front Wheel Angle: %f\n.", front_wheels_angle);
		ostream << vjoy_angle << "," << steering_ratio << "," << front_wheels_angle << std::endl;
	}
	ostream.close();
	//cv::waitKey(0);

}

