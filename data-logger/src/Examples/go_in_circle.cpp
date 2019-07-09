/*
 * cv_viewer.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */

#include "f1_datalogger/f1_datalogger.h"
 //#include "image_logging/utils/screencapture_lite_utils.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include "f1_datalogger/udp_logging/common/multi_threaded_udp_handler.h"
#include <boost/program_options.hpp>
#include "f1_datalogger/controllers/vjoy_interface.h"
#include <filesystem>
namespace fs = std::filesystem;
namespace po = boost::program_options;
void exit_with_help(po::options_description& desc)
{
	std::stringstream ss;
	ss << desc << std::endl;
	std::printf("%s", ss.str().c_str());
	exit(0); // @suppress("Invalid arguments")
}
int main(int argc, char** argv)
{
	std::string search_string, output_folder;
	double throttle, steering;
	unsigned int delay_time;


	po::options_description desc("F1 Datalogger Multithreaded Capture. Command line arguments are as follows");
	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("steering_angle,a", po::value<double>(&steering)->required(), "Steering angle to set.")
			("search_string,s", po::value<std::string>(&search_string)->default_value("2017"), "Search string for application to capture")
			("throttle,t", po::value<double>(&throttle)->default_value(.15), "Throttle setpoint to use.")
			("delay_time,d", po::value<unsigned int>(&delay_time)->default_value(3000), "Number of milliseconds to wait before capturing.")
			("output_folder,f", po::value<std::string>(&output_folder)->default_value("captured_data"), "UDP Output Folder")
			;
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
		if (vm.find("help") != vm.end()) {
			exit_with_help(desc);
		}
	}
	catch (boost::exception& e)
	{
		exit_with_help(desc);
	}

	std::shared_ptr<deepf1::IF1FrameGrabHandler> frame_handler;
	std::shared_ptr<deepf1::MultiThreadedUDPHandler> udp_handler(new deepf1::MultiThreadedUDPHandler(output_folder, 3, true));
	std::shared_ptr<deepf1::F1DataLogger> dl(new deepf1::F1DataLogger(search_string));
	std::cout << "Created DataLogger" << std::endl;


	std::string inp;
	deepf1::VJoyInterface vjoy(1);
	std::this_thread::sleep_for(std::chrono::milliseconds(1500));
	deepf1::F1ControlCommand command;
	//command.steering = 0.0;
	//command.throttle = 0.0;
	//vjoy.setCommands(command);
	std::cout << "Enter anything to start capture" << std::endl;
	std::cin >> inp;
	command.steering = steering;
	command.throttle = throttle;
	vjoy.setCommands(command);
	std::this_thread::sleep_for(std::chrono::milliseconds(delay_time));

	dl->start(30.0, udp_handler, frame_handler);

	std::cout << "Capturing data. Enter any key to end " << std::endl;
	std::cin >> inp;
	command.steering = 0.0;
	command.throttle = 0.0;
	vjoy.setCommands(command);
	udp_handler->stop();
	udp_handler->join();
}

