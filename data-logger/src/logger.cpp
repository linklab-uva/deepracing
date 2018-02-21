#pragma once
#include "deepf1_gsoap_conversions/gsoap_conversions.h"
#include <stdio.h>
#include <boost/timer/timer.hpp>
#include "simple_udp_listener.h"
#include "simple_screen_listener.h"
#include <iostream>
#include <exception>
#include <thread>
#include <string>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <memory>
#define BUFLEN 1289   //Max length of buffer
#define PORT 20777   //The port on which to listen for incoming data
#define DEFAULT_MAX_UDP_FRAMES 2000
#define DEFAULT_MAX_IMAGE_FRAMES 10
namespace po = boost::program_options;
namespace fs = boost::filesystem;
//using namespace deepf1;


	void cleanup_soap(soap* soap);

	
	void writeToFiles(const std::string& dir,
		const std::vector<deepf1::timestamped_image_data_t>& screen_data,
		const std::vector<deepf1::timestamped_udp_data>& udp_data);
	bool udp_data_comparator(const deepf1::timestamped_udp_data& a, const deepf1::timestamped_udp_data& b);
	deepf1::timestamped_udp_data find_closest_value(std::vector<deepf1::timestamped_udp_data>& udp_dataz,
		const boost::timer::cpu_times& timestamp);


int main(int argc, char** argv) {

	unsigned int udp_len, image_len;
	float capture_x, capture_y, capture_width, capture_height;
	int monitor_number;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "Hello World!!!")
		("udp_frames", po::value<unsigned int>(&udp_len)->default_value(100), "How many frames of game data to capture")
		("screen_frames", po::value<unsigned int>(&image_len)->default_value(100), "How many frames of screencap data to capture")
		("monitor_number", po::value<int>(&monitor_number)->default_value(1), "Monitor # to capture")
		("capture_x", po::value<float>(&capture_x)->default_value(0), "x coordinate for origin of capture area")
		("capture_y", po::value<float>(&capture_y)->default_value(0), "y coordinate for origin of capture area")
		("capture_width", po::value<float>(&capture_width)->default_value(100), "Width of capture area")
		("capture_height", po::value<float>(&capture_height)->default_value(100), "height of capture area")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	cv::Rect2d capture_area;
	if (capture_width > 0.0 && capture_height > 0.0) {
		capture_area = cv::Rect2d(capture_x, capture_y, capture_width, capture_height);
	}

	std::shared_ptr<const boost::timer::cpu_timer> timer(new boost::timer::cpu_timer);


	deepf1::simple_udp_listener udp_listener(timer, udp_len);
	std::function<void ()> udp_worker = std::bind(&deepf1::simple_udp_listener::listen, &udp_listener);
	std::thread udp_thread(udp_worker);

	deepf1::simple_screen_listener screen_listener(timer, capture_area, monitor_number, image_len);
	std::function<void()> screen_worker = std::bind(&deepf1::simple_screen_listener::listen, &screen_listener);
	std::thread screen_thread(screen_worker);

	std::string window;
	//std::getline(std::cin,window);
	udp_thread.join();
	screen_thread.join();
	std::vector<deepf1::timestamped_image_data_t> screen_data = screen_listener.get_data();
	std::vector<deepf1::timestamped_udp_data_t> udp_data = udp_listener.get_data();
	printf("screen timestamp at start: %llx \n", screen_data.at(0).timestamp.wall);
	printf("udp timestamp at start: %llx \n", udp_data.at(0).timestamp.wall);

	printf("screen timestamp half-way through: %llx \n", screen_data.at(screen_data.size() / 2).timestamp.wall);
	printf("udp timestamp half-way through: %llx \n", udp_data.at(udp_data.size() / 2).timestamp.wall);
/*
	cv::namedWindow(window);
	cv::imshow(window,*(data[image_len / 2].image));
	cv::waitKey();
	*/
//	deepf1::timestamped_udp_data fake_data;
//	fake_data.timestamp = data[image_len / 2].timestamp;
//	std::vector<deepf1::timestamped_udp_data>::iterator to_comp = std::lower_bound(udp_dataz.begin(), udp_dataz.end(), fake_data, udp_data_comparator);
	
//	printf("Value is: %f", udp_listener.get_data()[10].data.m_steer);
	//boost::function<void (const boost::timer::cpu_timer&, timestamped_udp_data_t[], unsigned int)> f1 = boost::bind(&udp_worker, (const boost::timer::cpu_timer) timer, udp_dataz, MAX_FRAMES);
	//
	//boost::thread screencap_trhead(boost::bind(&::screencap_worker, image_dataz, timer, MAX_FRAMES));

	return 0;
}
	void cleanup_soap(soap* soap)
	{
		// Delete instances
		soap_destroy(soap);
		// Delete data
		soap_end(soap);
		// Free soap struct engine context
		soap_free(soap);
	}
	bool udp_data_comparator(const deepf1::timestamped_udp_data& a, const deepf1::timestamped_udp_data& b) {
		return a.timestamp.wall < b.timestamp.wall;
	}
	deepf1::timestamped_udp_data find_closest_value(std::vector<deepf1::timestamped_udp_data>& udp_dataz,
		const boost::timer::cpu_times& timestamp) {
		deepf1::timestamped_udp_data fake_data;
		fake_data.timestamp = boost::timer::cpu_times(timestamp);
		std::vector<deepf1::timestamped_udp_data>::iterator to_comp = ::std::lower_bound(udp_dataz.begin(), udp_dataz.end(), fake_data, udp_data_comparator);
		if (to_comp == udp_dataz.begin())
		{
			return deepf1::timestamped_udp_data(udp_dataz[0]);
		}
		if (to_comp == udp_dataz.end())
		{
			return deepf1::timestamped_udp_data(udp_dataz[udp_dataz.size() - 1]);
		}
		long val = to_comp->timestamp.wall;
		long val_before = --to_comp->timestamp.wall;
		if (std::abs(timestamp.wall - val_before) < std::abs(timestamp.wall - val)) {
			return deepf1::timestamped_udp_data(*to_comp);
		}
		return deepf1::timestamped_udp_data(*(++to_comp));
	}
	void writeToFiles(const std::string& dir,
		const std::vector<deepf1::timestamped_image_data_t>& screen_data,
		const std::vector<deepf1::timestamped_udp_data>& udp_data) {
		soap* soap = soap_new();
		deepf1_gsoap_conversions::gsoap_conversions convert(soap);
		::deepf1_gsoap::UDPPacket* pack = convert.convert_to_gsoap(*(udp_data[0].data));

		cleanup_soap(soap);

	}
