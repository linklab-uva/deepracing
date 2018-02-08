
#include <Windows.h>
#include <stdio.h>
#include <boost/timer/timer.hpp>
#include "simple_udp_listener.h"
#include "simple_screen_listener.h"
#include <iostream>
#include <exception>
#include <thread>
#include <string>
#include <functional>
#include <boost/program_options.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#define BUFLEN 1289   //Max length of buffer
#define PORT 20777   //The port on which to listen for incoming data
#define DEFAULT_MAX_UDP_FRAMES 2000
#define DEFAULT_MAX_IMAGE_FRAMES 10
namespace po = boost::program_options;
int main(int argc, char** argv) {

	unsigned int udp_len, image_len;
	int monitor_number;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "Hello World!!!")
		("udp_frames", po::value<unsigned int>(&udp_len)->default_value(1000), "How many frames of game data to capture")
		("screen_frames", po::value<unsigned int>(&image_len)->default_value(250), "How many frames of screencap data to capture")
		("monitor_number", po::value<int>(&monitor_number)->default_value(1), "Monitor # to capture")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	/*
	deepf1::simple_screen_listener screen_listener(timer2);
	screen_listener.listen();
	*/
	boost::shared_ptr<const boost::timer::cpu_timer> timer(new boost::timer::cpu_timer);
	deepf1::simple_udp_listener udp_listener(timer, udp_len);
	std::function<void ()> udp_worker = std::bind(&deepf1::simple_udp_listener::listen, &udp_listener);
	std::thread udp_thread(udp_worker);

	deepf1::simple_screen_listener screen_listener(timer, monitor_number, image_len);
	std::function<void()> screen_worker = std::bind(&deepf1::simple_screen_listener::listen, &screen_listener);
	std::thread screen_thread(screen_worker);

	std::string window;
	std::getline(std::cin,window);
	deepf1::timestamped_image_data_t* data = screen_listener.get_data();
	screen_thread.join();
	cv::namedWindow(window);
	cv::imshow(window,*(data[image_len / 2].image));
	cv::waitKey();
	udp_thread.join();
	return 0;
//	printf("Value is: %f", udp_listener.get_data()[10].data.m_steer);
	//boost::function<void (const boost::timer::cpu_timer&, timestamped_udp_data_t[], unsigned int)> f1 = boost::bind(&udp_worker, (const boost::timer::cpu_timer) timer, udp_dataz, MAX_FRAMES);
	//
	//boost::thread screencap_trhead(boost::bind(&::screencap_worker, image_dataz, timer, MAX_FRAMES));

	return 0;
}