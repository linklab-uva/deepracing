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
#include "f1_datalogger/image_logging/common/multi_threaded_framegrab_handler.h"
#include "f1_datalogger/udp_logging/common/multi_threaded_udp_handler.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/math/special_functions.hpp>
namespace po = boost::program_options;
namespace fs = boost::filesystem;
void exit_with_help(po::options_description& desc)
{
	std::stringstream ss;
	ss << desc << std::endl;
	std::printf("%s", ss.str().c_str());
	exit(0); // @suppress("Invalid arguments")
}
int main(int argc, char** argv)
{
	using namespace deepf1;
	std::string search_string, image_folder, udp_folder, config_file, driver_name, track_name;
	unsigned int image_threads, udp_threads;
	float image_capture_frequency, initial_delay_time;
	double sine_frequency, throttle_val;


	po::options_description desc("F1 Datalogger Multithreaded Capture. Command line arguments are as follows");
	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("config_file,c", po::value<std::string>(&config_file)->required(), "Configuration file to load")
			("throttle_val,t", po::value<double>(&throttle_val)->required(), "Throttle value to set [-1,1]")
			("sine_frequency,f", po::value<double>(&sine_frequency)->required(), "Frequency of the sine wave to generate")
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
	std::cout << "Loading config file" << std::endl;
	YAML::Node config_node = YAML::LoadFile(config_file);
	std::cout << "Using the following config information:" << std::endl << config_node << std::endl;

	search_string = config_node["search_string"].as<std::string>();
	image_folder = config_node["images_folder"].as<std::string>();
	udp_folder = config_node["udp_folder"].as<std::string>();
	driver_name = config_node["driver_name"].as<std::string>();
	track_name = config_node["track_name"].as<std::string>();
	udp_threads = config_node["udp_threads"].as<unsigned int>();
	image_threads = config_node["image_threads"].as<unsigned int>();
	image_capture_frequency = config_node["image_capture_frequency"].as<float>();
	initial_delay_time = config_node["initial_delay_time"].as<float>();

	std::cout << "Creating handlers" << std::endl;


	std::string actual_image_folder = image_folder + "_throttle" + std::to_string(throttle_val) + "_frequency" + std::to_string(sine_frequency);
	std::shared_ptr<deepf1::MultiThreadedFrameGrabHandler> frame_handler(new deepf1::MultiThreadedFrameGrabHandler(actual_image_folder, image_threads, true));


	std::string actual_udp_folder = udp_folder + "_throttle" + std::to_string(throttle_val) + "_frequency" + std::to_string(sine_frequency);
	std::shared_ptr<deepf1::MultiThreadedUDPHandler> udp_handler(new deepf1::MultiThreadedUDPHandler(actual_udp_folder, udp_threads, true));

	std::ofstream configout((fs::path(actual_udp_folder) / fs::path("config.yaml")).string());
	configout << config_node << std::endl;
	configout.flush();
	configout.close();




	std::cout << "Creating DataLogger" << std::endl;
	std::shared_ptr<deepf1::F1DataLogger> dl(new deepf1::F1DataLogger(search_string, frame_handler, udp_handler));
	std::cout << "Created DataLogger" << std::endl;

	vjoy_plusplus::vJoy vjoy(1);
	vjoy_plusplus::JoystickPosition joystick_value;
	joystick_value.lButtons = 0x00000000;
	unsigned int min = vjoy_plusplus::vJoy::minAxisvalue(), max = vjoy_plusplus::vJoy::maxAxisvalue();
	unsigned int middle = (min + max) / 2;
	joystick_value.wAxisX = 0;
	joystick_value.wAxisY = 0;
	joystick_value.wAxisXRot = 0;
	joystick_value.wAxisYRot = 0;
	vjoy.update(joystick_value);
	deepf1::F1DataLogger::countdown(3, "Generating Sine Waves in");
	//dl->stop();
	double t = 0.0;
	double currentSteering = 0.0;
	double fake_zero = 0.0;
	double positive_deadband = fake_zero, negative_deadband = -fake_zero;
	double max_vjoysteer = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoythrottle = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoybrake = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	double middle_vjoysteer = max_vjoysteer / 2.0;
	std::chrono::high_resolution_clock clock;
	double twopi = 2.0*boost::math::constants::pi<double>();

	joystick_value.wAxisXRot = std::round(max_vjoythrottle*throttle_val);
	vjoy.update(joystick_value);
	dl->start();
	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::time_point(clock.now());
	double maxt = 10.0;
	while (t < 10.0)
	{
		t = 1E-6*((double)(std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - begin).count()));
		currentSteering = std::sin(t*twopi*sine_frequency);
		if (currentSteering > positive_deadband)
		{
			joystick_value.wAxisX = std::round(max_vjoysteer*currentSteering);
			joystick_value.wAxisY = 0;
		}
		else if (currentSteering < negative_deadband)
		{
			joystick_value.wAxisX = 0;
			joystick_value.wAxisY = std::round(max_vjoysteer * std::abs(currentSteering));
		}
		else
		{
			joystick_value.wAxisX = 0;
			joystick_value.wAxisY = 0;
		}
		vjoy.update(joystick_value);
	}
	joystick_value.wAxisXRot = 0;
	vjoy.update(joystick_value);
	while (t < 1.5*maxt)
	{
		t = 1E-6*((double)(std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - begin).count()));

		currentSteering = boost::math::sin_pi(2 * sine_frequency * t);
		if (currentSteering > positive_deadband)
		{
			joystick_value.wAxisX = std::round(max_vjoysteer*currentSteering);
			joystick_value.wAxisY = 0;
		}
		else if (currentSteering < negative_deadband)
		{
			joystick_value.wAxisX = 0;
			joystick_value.wAxisY = std::round(max_vjoysteer * std::abs(currentSteering));
		}
		else
		{
			joystick_value.wAxisX = 0;
			joystick_value.wAxisY = 0;
		}
		vjoy.update(joystick_value);
	}
	joystick_value.wAxisX = 0;
	joystick_value.wAxisY = 0;
	joystick_value.wAxisXRot = 0;
	joystick_value.wAxisYRot = 0;
	vjoy.update(joystick_value);
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

	frame_handler->stop();
	udp_handler->stop();
	frame_handler->join();
	udp_handler->join();

}

