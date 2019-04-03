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
	return a.timestamp() < b.timestamp();
}
class ReplayDataset_DataGrabHandler : public deepf1::IF1DatagrabHandler
{
public:
	ReplayDataset_DataGrabHandler(std::vector<deepf1::protobuf::TimestampedUDPData>& data, std::shared_ptr<vjoy_plusplus::vJoy> vjoy)
	{
		this->vjoy = vjoy;
		std::sort(data.begin(), data.end(), sortByTimestamp);
		std::cout << "Got a dataset with " << data.size() << " elements." << std::endl;
		for (unsigned int i = 0 ; i < data.size() ; i ++)
		{
			deepf1::protobuf::TimestampedUDPData datapoint = data.at(i);
			std::printf("Tag %u has Steering: %f, Throttle: %f, Brake: %f. Lap Time: %f\n", i, datapoint.udp_packet().m_steer(), datapoint.udp_packet().m_throttle(), datapoint.udp_packet().m_brake(), datapoint.udp_packet().m_laptime());
			if (datapoint.udp_packet().m_laptime() < 1E-4)
			{
				std::printf("Found the starting packet at %u with Steering: %f, Throttle: %f, Brake: %f. Lap Time: %f\n", i, datapoint.udp_packet().m_steer(), datapoint.udp_packet().m_throttle(), datapoint.udp_packet().m_brake(), datapoint.udp_packet().m_laptime());
				std::vector<deepf1::protobuf::TimestampedUDPData>::const_iterator first = data.begin() + i;
				std::vector<deepf1::protobuf::TimestampedUDPData>::const_iterator last = data.end();
				dataz_ = std::vector<deepf1::protobuf::TimestampedUDPData>(first, last);
				break;
			}
		}
		std::cout << "Extracted with " << dataz_.size() << " elements from dataset." << std::endl;
		for (unsigned int i = 0; i < dataz_.size(); i++)
		{	
			laptimes.push_back(dataz_.at(i).udp_packet().m_laptime());
			steering.push_back(dataz_.at(i).udp_packet().m_steer());
			throttle.push_back(dataz_.at(i).udp_packet().m_throttle());
			brake.push_back(dataz_.at(i).udp_packet().m_brake());
		}

		//std::string first_element;
		//google::protobuf::util::JsonOptions opshinz;
		//opshinz.always_print_primitive_fields = true;
		//opshinz.add_whitespace = true;
		//google::protobuf::util::MessageToJsonString(dataz_.at(0), &first_element, opshinz);
		//std::cout << first_element << std::endl;
	}
	bool isReady() override
	{
		return true;
	}
	void handleData(const deepf1::TimestampedUDPData& data) override
	{
		current_packet_ = data;
		std::vector<double>::iterator up;
		up = std::upper_bound(laptimes.begin(), laptimes.end(), data.data.m_lapTime);
		unsigned int idx = (up - laptimes.begin());
		double t2 = laptimes[idx];
		if (t2 >= 0.5)
		{
			double t1 = laptimes[idx-1];
			double DT = t2 - t1;
			double steer2 = steering[idx];
			double steer1 = steering[idx-1];
			double throttle2 = throttle[idx];
			double throttle1 = throttle[idx - 1];
			double brake2 = brake[idx];
			double brake1 = brake[idx - 1];
			double steer_slope = (steer2 - steer1) / DT;
			double throttle_slope = (throttle2 - throttle1) / DT;
			double brake_slope = (brake2 - brake1) / DT;
			double dt = data.data.m_lapTime - t1;
			double steer_command = steer1 + steer_slope * dt;
			if (std::abs(steer_command) < 1E-3)
			{
				steer_command = 0.0;
			}
			double throttle_command = throttle1 + throttle_slope * dt;
			if (throttle_command < 1E-3)
			{
				throttle_command = 0.0;
			}
			double brake_command = brake1 + brake_slope * dt;
			if (brake_command < 1E-3)
			{
				brake_command = 0.0;
			}
			js.wAxisY = (unsigned int)std::round(-16383.813867*steer_command + 16383.630437);
			js.wAxisZ = (unsigned int)(max_vjoythrottle*throttle_command);
			js.wAxisZRot = (unsigned int)(max_vjoybrake*brake_command);
			vjoy->update(js);
			//std::printf("Applying a command of Steering: %f, Throttle: %f, Brake: %f\n", steer_command, throttle_command, brake_command);

		}
	}
	void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override
	{
		max_vjoysteer = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoythrottle = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoybrake = (double)vjoy_plusplus::vJoy::maxAxisvalue();
		this->begin = begin;
	}
	deepf1::TimestampedUDPData getCurrentPacket()
	{
		return current_packet_;
	}
	std::chrono::high_resolution_clock::time_point getBegin()
	{
		return begin;
	}
private:
	std::chrono::high_resolution_clock::time_point begin;
	deepf1::TimestampedUDPData current_packet_;
	std::vector<double> laptimes, steering, throttle, brake;
	std::vector<deepf1::protobuf::TimestampedUDPData> dataz_;
	std::shared_ptr<vjoy_plusplus::vJoy> vjoy;
	double max_vjoysteer, max_vjoythrottle, max_vjoybrake;
	vjoy_plusplus::JoystickPosition js;
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
	std::shared_ptr<vjoy_plusplus::vJoy> vjoy(new vjoy_plusplus::vJoy(1));
	vjoy_plusplus::JoystickPosition iReport;
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (unsigned int)std::round(0.5*(double)(min + max));
	iReport.lButtons = 0x00000000;
	iReport.wAxisY = middle;
	iReport.wAxisZ = 0;
	iReport.wAxisZRot = 0;
	vjoy->update(iReport);
	std::vector<deepf1::protobuf::TimestampedUDPData> data = deepf1::post_processing::PostProcessingUtils::parseUDPDirectory(*dir);
	std::shared_ptr<ReplayDataset_FrameGrabHandler> image_handler(new ReplayDataset_FrameGrabHandler());
	std::shared_ptr<ReplayDataset_DataGrabHandler> udp_handler(new ReplayDataset_DataGrabHandler(data, vjoy));
	deepf1::F1DataLogger dl(*search, image_handler, udp_handler);
	dl.start();
	countdown(3, "Starting in");
	std::string s;
	std::cin >> s;
	//cv::waitKey(0);

}

