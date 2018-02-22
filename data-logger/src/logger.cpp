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
#include <sstream>
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
		std::vector<deepf1::timestamped_udp_data>& udp_data,
		const long& max_delta);
	bool udp_data_comparator(const deepf1::timestamped_udp_data& a, const deepf1::timestamped_udp_data& b);
	deepf1::timestamped_udp_data find_closest_value(std::vector<deepf1::timestamped_udp_data>& udp_dataz,
		const boost::timer::cpu_times& timestamp);
}
int main(int argc, char** argv) {

	unsigned int udp_len, image_len;
	float capture_x, capture_y, capture_width, capture_height;
	int monitor_number;
	long max_delta;
	std::string data_directory;
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
		("max_delta", po::value<long>(&max_delta)->default_value(20), "Maximum difference in timestamp (in milliseconds) to allow for associating data to an image")
		("data_directory", po::value<std::string>(&data_directory)->default_value(std::string("data")), "Top-level directory to place the annotations & images.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	

	std::shared_ptr<const boost::timer::cpu_timer> timer(new boost::timer::cpu_timer);


	deepf1::simple_udp_listener udp_listener(timer, udp_len);
	std::function<void ()> udp_worker = std::bind(&deepf1::simple_udp_listener::listen, &udp_listener);
	std::thread udp_thread(udp_worker);

	cv::Rect2d capture_area = cv::Rect2d(capture_x, capture_y, capture_width, capture_height);
	deepf1::simple_screen_listener screen_listener(timer, capture_area, monitor_number, image_len);
	std::function<void()> screen_worker = std::bind(&deepf1::simple_screen_listener::listen, &screen_listener);
	std::thread screen_thread(screen_worker);


	udp_thread.join();
	screen_thread.join();
	std::vector<deepf1::timestamped_image_data_t> screen_data = screen_listener.get_data();
	std::vector<deepf1::timestamped_udp_data_t> udp_data = udp_listener.get_data();
	printf("screen timestamp at start: %llx \n", screen_data.at(0).timestamp.wall);
	printf("udp timestamp at start: %llx \n", udp_data.at(0).timestamp.wall);

	printf("screen timestamp half-way through: %llx \n", screen_data.at(screen_data.size() / 2).timestamp.wall);
	printf("udp timestamp half-way through: %llx \n", udp_data.at(udp_data.size() / 2).timestamp.wall);
	writeToFiles(data_directory, screen_data, udp_data, max_delta);

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
		std::vector<deepf1::timestamped_udp_data>& udp_data,
		const long& max_delta) {
		fs::create_directory(fs::path(dir));
		fs::path annotations_dir = fs::path(dir)/ fs::path("annotations");
		fs::create_directory(annotations_dir);
		fs::path images_dir = fs::path(dir)/fs::path("images");
		fs::create_directory(images_dir);
		soap* soap = soap_new(SOAP_XML_INDENT);
		deepf1_gsoap_conversions::gsoap_conversions convert(soap);
		unsigned long point_number = 1;
		for (auto it = screen_data.begin(); it != screen_data.end(); it++) {

			deepf1::timestamped_udp_data udp_tag = find_closest_value(udp_data, it->timestamp);
			long delta = (std::abs(udp_tag.timestamp.wall - it->timestamp.wall))/1E6;
			if (delta < max_delta) {
				std::printf("Associating an image with timestamp %lld to upd packet with timestamp %lld\n", it->timestamp.wall, udp_tag.timestamp.wall);
			}
			else {
				std::printf("Discarding image because the closest udp data is %lld milliseconds away", delta);
				continue;

			}

			::deepf1_gsoap::ground_truth_sample * ground_truth = deepf1_gsoap::soap_new_ground_truth_sample(soap);
			::deepf1_gsoap::UDPPacket* pack = convert.convert_to_gsoap(*(udp_tag.data));
			ground_truth->sample = *pack;


			std::stringstream image_ss;
			image_ss << "image_" << point_number << ".jpg";
			fs::path image_path = images_dir / fs::path(image_ss.str());
			cv::imwrite(image_path.string(), *(it->image));
			ground_truth->image_file = image_ss.str();

			std::stringstream annotation_ss;
			annotation_ss << "data_point_" << point_number << ".xml";
			++point_number;
			fs::path annotation_path = annotations_dir / fs::path(annotation_ss.str());
		
			soap->os = new std::fstream(annotation_path.string(), std::fstream::out);
			deepf1_gsoap::soap_write_ground_truth_sample(soap, ground_truth);
			((std::fstream*)(soap->os))->close();
			delete (soap->os);
			soap_destroy(soap);

		}


		cleanup_soap(soap);

	}
}