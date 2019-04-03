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
class ReplayDataset_DataGrabHandler : public deepf1::IF1DatagrabHandler
{
public:
	ReplayDataset_DataGrabHandler()
	{

	}
	bool isReady() override
	{
		return true;
	}
	void handleData(const deepf1::TimestampedUDPData& data) override
	{
		current_packet_ = data;
	}
	void init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin) override
	{
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
	po::options_description desc("Allowed Options");

	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("search_string,s", po::value<std::string>(search.get())->default_value("2017"), "Search string to find the window name for F1 2017")
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
	std::shared_ptr<ReplayDataset_FrameGrabHandler> image_handler(new ReplayDataset_FrameGrabHandler());
	std::shared_ptr<ReplayDataset_DataGrabHandler> udp_handler(new ReplayDataset_DataGrabHandler());
	deepf1::F1DataLogger dl(*search, image_handler, udp_handler);
	dl.start();
	std::unique_ptr<vjoy_plusplus::vJoy> vjoy(new vjoy_plusplus::vJoy(1));
	vjoy_plusplus::JoystickPosition iReport;
	iReport.lButtons = 0x00000000;
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (unsigned int)std::round(0.5*(double)(min + max));
	iReport.wAxisY = middle;
	iReport.wAxisZ = 0;
	iReport.wAxisZRot = 0;
	vjoy->update(iReport);
	countdown(3, "Starting in");
	//cv::waitKey(0);

}

