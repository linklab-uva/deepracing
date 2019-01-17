#include <stdio.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <f1_datalogger/post_processing/post_processing_utils.h>
#include <algorithm>
#include <iostream>
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
	std::string config_file;


	po::options_description desc("F1 Datalogger Multithreaded Capture. Command line arguments are as follows");
	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("config_file,f", po::value<std::string>(&config_file)->required(), "Configuration file to load")
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

	std::string image_folder = config_node["images_folder"].as<std::string>();
	std::string udp_folder = config_node["udp_folder"].as<std::string>();

	std::vector<deepf1::protobuf::F1UDPData> udp_points = deepf1::post_processing::PostProcessingUtils::parseUDPDirectory(udp_folder);
	std::vector<deepf1::protobuf::TimestampedImage> image_points = deepf1::post_processing::PostProcessingUtils::parseImageDirectory(image_folder);


	std::printf("Got %lu udp data points.\n", udp_points.size());
	std::printf("Got %lu image data points.\n", image_points.size());
	std::vector<deepf1::protobuf::LabeledImage> labeled_images = 
		 deepf1::post_processing::PostProcessingUtils::labelImages(udp_points,  image_points, 3);


}