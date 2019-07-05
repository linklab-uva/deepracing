#include <stdio.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <f1_datalogger/post_processing/post_processing_utils.h>
#include <algorithm>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <google/protobuf/util/json_util.h>
#include <opencv2/imgproc.hpp>
namespace po = boost::program_options;
namespace fs = std::filesystem;
void exit_with_help(po::options_description& desc)
{
	std::stringstream ss;
	ss << desc << std::endl;
	std::printf("%s", ss.str().c_str());
	exit(0); // @suppress("Invalid arguments")
}
int main(int argc, char** argv)
{
	std::string config_file, label_dir;


	po::options_description desc("F1 Datalogger Multithreaded Capture. Command line arguments are as follows");
	try {
		desc.add_options()
			("help,h", "Displays options and exits")
			("config_file,f", po::value<std::string>(&config_file)->required(), "Configuration file to load")
			("label_directory,l", po::value<std::string>(&label_dir)->default_value("image_labels"), "Configuration file to load")
			;
		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
		if (vm.find("help") != vm.end())
		{
			exit_with_help(desc);
		}
	}
	catch (boost::exception& e)
	{
		exit_with_help(desc);
	}
	fs::path labels_path(label_dir);
	if ( !fs::is_directory(labels_path) )
	{
		fs::create_directories(labels_path);
	}
	std::cout << "Loading config file" << std::endl;
	YAML::Node config_node = YAML::LoadFile(config_file);

	std::string image_folder = config_node["images_folder"].as<std::string>();
	std::string udp_folder = config_node["udp_folder"].as<std::string>();

	std::vector<deepf1::protobuf::TimestampedUDPData> udp_points = deepf1::post_processing::PostProcessingUtils::parseUDPDirectory(udp_folder);
	std::vector<deepf1::protobuf::TimestampedImage> image_points = deepf1::post_processing::PostProcessingUtils::parseImageDirectory(image_folder);


	std::printf("Got %lu udp data points.\n", (unsigned long)udp_points.size());
	std::printf("Got %lu image data points.\n", (unsigned long)image_points.size());
	unsigned int INTERPOLATION_ORDER = 2;
	std::vector<deepf1::protobuf::LabeledImage> labeled_images = 
		 deepf1::post_processing::PostProcessingUtils::labelImages(udp_points,  image_points, INTERPOLATION_ORDER);

	cv::namedWindow("image",cv::WINDOW_AUTOSIZE);
	float arrow_length = 100.0;
	for (unsigned int i = 0; i < labeled_images.size(); i ++)
	{
		deepf1::protobuf::LabeledImage labeled_image = labeled_images.at(i); 
/*
		if (labeled_image.image_file().empty())
		{
			continue;
		}

		printf("Image file in labeled image: %s", labeled_image.image_file().c_str());
		*/
		fs::path full_image_path = fs::path(image_folder) / fs::path(labeled_image.image_file());
		std::string json;

		google::protobuf::util::JsonOptions opshinz;
		opshinz.always_print_primitive_fields = true;
		opshinz.add_whitespace = true;
		google::protobuf::util::MessageToJsonString(labeled_image, &json, opshinz);
		/**/
		std::cout << "Label: " << json << std::endl;
		
		cv::Mat im_mat = cv::imread(full_image_path.string());
		cv::Mat arrow_mat(im_mat.rows, im_mat.cols, im_mat.type(), cv::Scalar(255,255,255));
		


		cv::Point p1(im_mat.cols/2, im_mat.rows/2);
		float real_angle = 2.0*((std::asin<float>(labeled_image.label().steering()).real()));
		
		float cos_ = arrow_length*(std::cos<float>(real_angle).real());
		float sin_ = arrow_length*(std::sin<float>(real_angle).real());
		cv::Point p2(p1.x - sin_, p1.y - cos_);
		cv::arrowedLine(im_mat, p1, p2, cv::Scalar(255));
		cv::imshow("image", im_mat);
		cv::waitKey(17);
	

		std::string pb_fn = "image_label_" + std::to_string(i) + ".pb";
		fs::path full_path = labels_path / fs::path(pb_fn);
		std::ofstream ostream;
		ostream.open(full_path.c_str() , std::fstream::out);
		labeled_image.SerializeToOstream(&ostream);
		ostream.close();

		fs::path json_path = labels_path / fs::path(pb_fn + ".json");
	//	std::string json;
	//	google::protobuf::util::MessageToJsonString(labeled_image, &json);
		ostream.open(json_path.c_str(), std::fstream::out);
		ostream << json;
	}


}