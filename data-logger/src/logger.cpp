#pragma once
#include "deepf1_gsoap_conversions/gsoap_conversions.h"
#include <stdio.h>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <exception>
#include <thread>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <memory>
#include <sstream>
#include "simple_udp_listener.h"
#include "simple_screen_listener.h"
#define BUFLEN 1289   //Max length of buffer
#define PORT 20777   //The port on which to listen for incoming data
#define DEFAULT_MAX_UDP_FRAMES 2000
#define DEFAULT_MAX_IMAGE_FRAMES 10
namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace deepf1{
	void cleanup_soap(soap* soap);

	
	void writeToFiles(const std::string& dir,
		std::vector<deepf1::timestamped_image_data_t>& screen_data,
		deepf1::simple_udp_listener& udp_listener);
	bool udp_data_comparator(const deepf1::timestamped_udp_data& a, const deepf1::timestamped_udp_data& b);
	deepf1::timestamped_udp_data find_closest_value(std::vector<deepf1::timestamped_udp_data>& udp_dataz,
		const boost::timer::cpu_times& timestamp);
}
/*
-x 100
-y 250
-w 1600
-h 375
seem to be good values.
*/
int main(int argc, char** argv) {

	unsigned int udp_len, image_len;
	float capture_x, capture_y, capture_width, capture_height;
	int monitor_number;
	long sleep_time;
	unsigned short port_number;
	std::string data_directory, application_name;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "Displays options and exits")
		("udp_frames,u", po::value<unsigned int>(&udp_len)->default_value(10000), "How many frames of game data to capture")
		("port_number,p", po::value<unsigned short>(&port_number)->default_value(PORT), "Port number to listen for telemetry data on")
		("screen_frames,s", po::value<unsigned int>(&image_len)->default_value(15000), "How many frames of screencap data to capture")
		("capture_x,x", po::value<float>(&capture_x)->default_value(0), "x coordinate for origin of capture area in pixels")
		("capture_y,y", po::value<float>(&capture_y)->default_value(430), "y coordinate for origin of capture area pixels")
		("capture_width,w", po::value<float>(&capture_width)->default_value(2510), "Width of capture area pixels")
		("capture_height,h", po::value<float>(&capture_height)->default_value(200), "height of capture area pixels")
		("data_directory,d", po::value<std::string>(&data_directory)->default_value(std::string("data")), "Top-level directory to place the annotations & images.")
		("initial_sleep_time,i", po::value<long>(&sleep_time)->default_value(5000), "How many milliseconds to sleep before starting data recording.")
		("application_name,a", po::value<std::string>(&application_name)->default_value(std::string("")), "Name of the application to capture. If not set, defaults to the desktop window.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.find("help") != vm.end()) {
		std::stringstream ss;
		ss << "F1 Datalogger. Command line arguments are as follows:" << std::endl;
		desc.print(ss);
		std::printf("%s", ss.str().c_str());
		exit(0);
	}

	cv::Rect2d capture_area = cv::Rect2d(capture_x, capture_y, capture_width, capture_height);
	std::printf("Starting data capture in %lld milliseconds\n", sleep_time);
	Sleep(sleep_time);

	
	std::shared_ptr<const boost::timer::cpu_timer> timer(new boost::timer::cpu_timer);
	deepf1::simple_udp_listener udp_listener(timer, udp_len, port_number);
	deepf1::simple_screen_listener screen_listener(timer, capture_area, application_name, image_len);


	std::function<void()> udp_worker = std::bind(&deepf1::simple_udp_listener::listen, &udp_listener);
	Sleep(1500);
	std::function<void()> screen_worker = std::bind(&deepf1::simple_screen_listener::listen, &screen_listener);

	std::thread udp_thread(udp_worker);
	std::thread screen_thread(screen_worker);
	screen_thread.join();
	std::cout << "Screencapping done" << std::endl;
	udp_listener.stop();
	udp_thread.join();
	std::vector<deepf1::timestamped_image_data_t> screen_data = screen_listener.get_data();
	std::vector<deepf1::timestamped_udp_data_t> udp_data = udp_listener.get_data();
	std::printf("Started receiving packets at timestamp: %lld \n", udp_listener.get_collection_start().wall);
	writeToFiles(data_directory, screen_data, udp_listener);

	return 0;
}
namespace deepf1 {

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
		long val_before = (--to_comp)->timestamp.wall;
		if (std::abs(timestamp.wall - val_before) < std::abs(timestamp.wall - val)) {
			return deepf1::timestamped_udp_data(*to_comp);
		}
		return deepf1::timestamped_udp_data(*(++to_comp));
	}
	void writeToFiles(const std::string& dir,
		std::vector<deepf1::timestamped_image_data_t>& screen_data,
		deepf1::simple_udp_listener& udp_listener) {
		std::vector<deepf1::timestamped_udp_data> udp_data = udp_listener.get_data();
		fs::path root_dir = fs::path(dir);
		fs::create_directory(root_dir);
		fs::path raw_annotations_dir = root_dir / fs::path("raw_annotations");
		fs::create_directory(raw_annotations_dir);
		fs::path raw_images_dir = root_dir / fs::path("raw_images");
		fs::create_directory(raw_images_dir);

		fs::path raw_images_timestamps = raw_images_dir / fs::path("raw_image_timestamps.csv");
		soap* soap = soap_new(SOAP_XML_INDENT);
		deepf1_gsoap_conversions::gsoap_conversions convert(soap);
		unsigned long point_number = 1;
		unsigned long raw_point_number = 1;
		std::shared_ptr<std::fstream> file_out;
		std::shared_ptr<std::ofstream> raw_image_timestamps_stream(new std::ofstream(raw_images_timestamps.string(), std::fstream::out));
		std::cout << "Writing " << screen_data.size() << " raw images to file." << std::endl;
		for (auto it = screen_data.begin(); it != screen_data.end(); it++) {
			
			std::stringstream raw_image_ss;
			raw_image_ss << "raw_image_" << raw_point_number << ".jpg";
			fs::path raw_image_path = raw_images_dir / fs::path(raw_image_ss.str());
			cv::imwrite(raw_image_path.string(), (it->image));
			(*raw_image_timestamps_stream) << raw_image_ss.str() << ", " << (it->timestamp.wall) << std::endl;
			raw_point_number++;


		}
		raw_point_number = 1;
		std::cout << "Writing " << udp_data.size() << " raw annotations to file." << std::endl;
		for (unsigned int i = 0; i < udp_data.size(); i++) {
			deepf1::timestamped_udp_data current_data = udp_data[i];

			::deepf1_gsoap::ground_truth_sample * ground_truth = deepf1_gsoap::soap_new_ground_truth_sample(soap);
			::deepf1_gsoap::UDPPacket* pack = convert.convert_to_gsoap(*(current_data.data));
			ground_truth->sample = *pack;
			ground_truth->timestamp = current_data.timestamp.wall;
			ground_truth->image_file = "NOT ASSOCIATED";

			std::stringstream annotation_ss;
			annotation_ss << "raw_data_point_" << raw_point_number << ".xml";
			++raw_point_number;
			fs::path annotation_path = raw_annotations_dir / fs::path(annotation_ss.str());
			file_out.reset(new std::fstream(annotation_path.string(), std::fstream::out));
			soap->os = file_out.get();
			deepf1_gsoap::soap_write_ground_truth_sample(soap, ground_truth);
			file_out->close();
		}
		cleanup_soap(soap);

	}
}