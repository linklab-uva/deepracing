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
#include "f1_datalogger/udp_logging/common/measurement_handler.h"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <f1_datalogger/controllers/pure_pursuit_controller.h>
namespace fs = boost::filesystem;
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
	std::string search_string, trackfile;
	double lookahead_gain, throttle;


	po::options_description desc("F1 Datalogger Multithreaded Capture. Command line arguments are as follows");
	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("search_string,s", po::value<std::string>(&search_string)->required(), "Search string for application to capture")
			("trackfile,f", po::value<std::string>(&trackfile)->required(), "Trackfile to read the raceline from.")
			("lookahead_gain,g", po::value<double>(&lookahead_gain)->required(), "Linear Lookahead gain for the pure pursuit controller")
			("throttle,t", po::value<double>(&throttle)->default_value(0.2), "Fixed throttle value to use.")
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

	std::shared_ptr<deepf1::MeasurementHandler> udp_handler(new deepf1::MeasurementHandler());
	std::shared_ptr<deepf1::IF1FrameGrabHandler> image_handler;

	deepf1::F1DataLogger dl(search_string, image_handler, udp_handler);
	dl.start();
	deepf1::PurePursuitController control(udp_handler,lookahead_gain,3.7,1.0,throttle);
	deepf1::F1DataLogger::countdown(3, "Running pure pursuit in ");
	control.run(trackfile);
}

