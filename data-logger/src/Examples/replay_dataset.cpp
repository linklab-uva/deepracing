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
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <f1_datalogger/post_processing/post_processing_utils.h>
#include <google/protobuf/util/json_util.h>
#include <boost/thread/barrier.hpp> 
namespace po = boost::program_options;
namespace fs = boost::filesystem;
void exit_with_help(po::options_description& desc)
{
	std::stringstream ss;
	ss << "F1 Replay Dataset. Command line arguments are as follows:" << std::endl;
	desc.print(ss);
	std::printf("%s", ss.str().c_str());
	exit(0);
}
void countdown(unsigned int seconds, std::string text = "")
{
	std::cout << text << std::endl;
	for (unsigned int i = seconds; i > 0; i--)
	{
		std::cout << i << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}
bool sortByTimestamp(const deepf1::protobuf::TimestampedUDPData& a, const deepf1::protobuf::TimestampedUDPData& b)
{
	return a.udp_packet().m_time() < b.udp_packet().m_time();
}
class ReplayDataset_DataGrabHandler : public deepf1::IF1DatagrabHandler
{
public:
	ReplayDataset_DataGrabHandler(boost::barrier& bar) :
		bar_(bar), waiting_(true)
	{
	}
	bool isReady() override
	{
		return true;
	}
	void handleData(const deepf1::TimestampedUDPData& data) override
	{
		if (waiting_ && data.data.m_lapTime<=1E-3)
		{
			bar_.wait();
			waiting_ = false;
		}
	}
	void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override
	{
		
		this->begin = begin;
	}
	std::chrono::high_resolution_clock::time_point getBegin()
	{
		return begin;
	}
private:
	std::chrono::high_resolution_clock::time_point begin;
	boost::barrier& bar_;
	bool waiting_;
};
class ReplayDataset_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
	ReplayDataset_FrameGrabHandler()
	{
	}
	virtual ~ReplayDataset_FrameGrabHandler()
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
		this->begin = begin;
	}
	std::chrono::high_resolution_clock::time_point getBegin()
	{
		return begin;
	}
private:
	std::chrono::high_resolution_clock::time_point begin;
};
void setControls(const std::vector<double>& laptimes, const std::vector<double>& steering,
	const std::vector<double>& throttle, const std::vector<double>& brake, 
	const double& time, const double& max_vjoy, vjoy_plusplus::JoystickPosition& js)
{
	
}
int main(int argc, char** argv)
{
	std::unique_ptr<std::string> search(new std::string);
	std::unique_ptr<std::string> dir(new std::string);
	po::options_description desc("Allowed Options");

	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("search_string,s", po::value<std::string>(search.get())->default_value("2017"), "Search string to find the window name for F1 2017")
			("data_dir,d", po::value<std::string>(dir.get())->required(), "Directory to look for stored UDP data")
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
	vjoy_plusplus::vJoy vjoy(1);
	vjoy_plusplus::JoystickPosition js;
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (unsigned int)std::round(0.5*(double)(min + max));
	js.lButtons = 0x00000000;
	js.wAxisX = 0;
	js.wAxisY = 0;
	js.wAxisXRot = 0;
	js.wAxisYRot = 0;
	vjoy.update(js);
	std::vector<deepf1::protobuf::TimestampedUDPData> data = deepf1::post_processing::PostProcessingUtils::parseUDPDirectory(*dir);
	std::vector<deepf1::protobuf::TimestampedUDPData> sorted_data;
	std::sort(data.begin(), data.end(), sortByTimestamp);
	std::cout << "Got a dataset with " << data.size() << " elements." << std::endl;
	for (unsigned int i = 0; i < data.size(); i++)
	{
		deepf1::protobuf::TimestampedUDPData datapoint = data.at(i);
		std::printf("Tag %u has Steering: %f, Throttle: %f, Brake: %f. Lap Time: %f\n", i, datapoint.udp_packet().m_steer(), datapoint.udp_packet().m_throttle(), datapoint.udp_packet().m_brake(), datapoint.udp_packet().m_laptime());
		if (datapoint.udp_packet().m_laptime() < 1E-4)
		{
			std::printf("Found the starting packet at %u with Steering: %f, Throttle: %f, Brake: %f. Lap Time: %f\n", i, datapoint.udp_packet().m_steer(), datapoint.udp_packet().m_throttle(), datapoint.udp_packet().m_brake(), datapoint.udp_packet().m_laptime());
			std::vector<deepf1::protobuf::TimestampedUDPData>::const_iterator first = data.begin() + i;
			std::vector<deepf1::protobuf::TimestampedUDPData>::const_iterator last = data.end();
			sorted_data = std::vector<deepf1::protobuf::TimestampedUDPData>(first, last);
			break;
		}
	}
	std::cout << "Extracted with " << sorted_data.size() << " elements from dataset." << std::endl;
	std::vector<double> laptimes, steering, throttle, brake;
	double timesteps = 3.0;
	for (unsigned int i = 1; i < sorted_data.size(); i++)
	{
		double currentTime = sorted_data.at(i - 1).udp_packet().m_laptime();
		double currentSteer = sorted_data.at(i - 1).udp_packet().m_steer();
		double currentThrottle = sorted_data.at(i - 1).udp_packet().m_throttle();
		double currentBrake = sorted_data.at(i - 1).udp_packet().m_brake();

		double timeUB = sorted_data.at(i).udp_packet().m_laptime();
		double steerUB = sorted_data.at(i).udp_packet().m_steer();
		double throttleUB = sorted_data.at(i).udp_packet().m_throttle();
		double brakeUB = sorted_data.at(i).udp_packet().m_brake();

		double DT;
		double throttle2;
		double throttle1;
		double brake2;
		double brake1;
		double steer_slope;
		double throttle_slope;
		double brake_slope;
		double steer_command;
		double throttle_command;
		double brake_command;
		laptimes.push_back(currentTime);
		steering.push_back(currentSteer);
		throttle.push_back(currentThrottle);
		brake.push_back(currentBrake);

		DT = timeUB - currentTime;
		steer_slope = (steerUB - currentSteer) / DT;
		throttle_slope = (throttleUB - currentThrottle) / DT;
		brake_slope = (brakeUB - currentBrake) / DT;
		double dt = DT / timesteps;
		for (double time = currentTime + dt; time < timeUB; time += dt)
		{
			double dt = time - currentTime;
			steer_command = currentSteer + steer_slope * dt;
		    throttle_command = currentThrottle + throttle_slope * dt;
			brake_command = currentBrake + brake_slope * dt;
			laptimes.push_back(time);
			steering.push_back(steer_command);
			throttle.push_back(throttle_command);
			brake.push_back(brake_command);
		}
	}
	double max_vjoysteer = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoythrottle = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoybrake = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	double middle_vjoysteer = max_vjoysteer / 2.0;
	std::chrono::high_resolution_clock clock;
	boost::barrier bar(2);
	std::shared_ptr<ReplayDataset_FrameGrabHandler> image_handler(new ReplayDataset_FrameGrabHandler());
	std::shared_ptr<ReplayDataset_DataGrabHandler> udp_handler(new ReplayDataset_DataGrabHandler(boost::ref(bar)));
	std::unique_ptr<deepf1::F1DataLogger> dl(new deepf1::F1DataLogger(*search, image_handler, udp_handler));
	dl->start();
	double time;
    int idx;
	time = 0.0;
	double maxtime = laptimes.back();
	bar.wait();
	//std::cout << "Got past the barrier" << std::endl;
	std::chrono::high_resolution_clock::time_point begin = clock.now();
	//dl.reset();
	//Best fit line is : y = -16383.813867*x + 16383.630437
	while (time< maxtime)
	{
		time = 1E-6*((double)(std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - begin).count()));
		idx = (std::upper_bound(laptimes.begin(), laptimes.end(), time) - laptimes.begin());
		//if (idx >= laptimes.size() - (unsigned int)timesteps - 1)
		//{
		//	break;
		//}
		if (steering[idx] > 1E-3)
		{
			js.wAxisX = (unsigned int)std::round(max_vjoysteer * steering[idx]);
			js.wAxisY = 0;
		}
		else if (steering[idx] < -(1E-3))
		{
			js.wAxisX = 0;
			js.wAxisY = (unsigned int)std::round(max_vjoysteer * -steering[idx]);
		}
		else
		{
			js.wAxisX = 0;
			js.wAxisY = 0;
		}
		js.wAxisXRot = (unsigned int)std::round(max_vjoythrottle*throttle[idx]);
		js.wAxisYRot = (unsigned int)std::round(max_vjoybrake*brake[idx]);
		vjoy.update(js);
	}
	std::cout << "Thanks for Playing!" << std::endl;
	js.wAxisX = 0;
	js.wAxisY = 0;
	js.wAxisXRot = 0;
	js.wAxisYRot = 0;
	vjoy.update(js);
	//cv::waitKey(0);

}

