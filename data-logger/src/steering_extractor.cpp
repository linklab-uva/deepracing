#include "deepf1_gsoap/deepf1_gsoapH.h"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <memory>
#include <iostream>
#include <fstream>
namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace deepf1 {

	void cleanup_soap(soap* gsoap);
}
int main(int argc, char** argv) {

	unsigned int interpolation_degree;
	std::string data_directory, annotation_prefix, output_file;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "Displays options and exits")
		("data_folder,f", po::value<std::string>(&data_directory)->default_value(std::string("data")), "Top-level folder to look for the original data dump.")
		("annotation_prefix,p", po::value<std::string>(&annotation_prefix)->default_value(std::string("data_point_")), "Prefix of the filename for each native annotation file. Each annotation is the prefix followed by a unique integer.")
		("output_file,o", po::value<std::string>(&output_file)->default_value(std::string("out.csv")), "Output file to dump the steering angles to.")
		("interpolation_degree,d", po::value<unsigned int>(&interpolation_degree)->default_value(0), "What degree of polynomial interpolation to use. It left at the default value of 0, no interpolation is done.")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.find("help") != vm.end()) {
		std::stringstream ss;
		ss << "F1 Datadump Steering Angle Extractor. Command line arguments are as follows:" << std::endl;
		desc.print(ss);
		std::printf("%s", ss.str().c_str());
		exit(0);
	}
	fs::path root_dir(data_directory);

	if (!fs::is_directory(root_dir)) {
		std::cerr << "ERROR: Root data directory " << root_dir.string() << "does not exist." << std::endl;
		exit(-1);
	}

	std::cout << "Found the data root directory" << std::endl;


	fs::path annotations_dir = root_dir / fs::path("annotations");

	if (!fs::is_directory(annotations_dir)) {
		std::cerr << "ERROR: Annotations directory " << root_dir.string() << "does not exist." << std::endl;
		exit(-1);
	}
	std::cout << "Found the annotations directory" << std::endl;

	soap* file_reader = soap_new();

	if(interpolation_degree ==0){
		std::shared_ptr<std::fstream> file_in;
		std::ofstream file_out(output_file,std::fstream::out);
		int index = 1;
		fs::path current_file = annotations_dir / fs::path(annotation_prefix + std::to_string(index) + ".xml");
		std::cout << "Checking for file: " << current_file.string() << std::endl;
		while (fs::exists(current_file)) {
			std::cout << "Processing file: " << current_file.string() << std::endl;
			file_in.reset(new std::fstream(current_file.string(),std::fstream::in));
			::deepf1_gsoap::ground_truth_sample * ground_truth = deepf1_gsoap::soap_new_ground_truth_sample(file_reader);
			file_reader->is = file_in.get();
			deepf1_gsoap::soap_read_ground_truth_sample(file_reader, ground_truth);
			file_out << ground_truth->image_file << "," << ground_truth->sample.m_steer << "," << ground_truth->timestamp << std::endl;


			current_file = annotations_dir / fs::path(annotation_prefix + std::to_string(++index) + ".xml");
			std::cout << "Checking for file: " << current_file.string() << std::endl;
		}
		std::cout << "Done. Each row of: " << output_file << " is <image_file>,<steering_angle>,<timestamp>" << std::endl;
	}
	deepf1::cleanup_soap(file_reader);
	return 0;
}
namespace deepf1 {

	void cleanup_soap(soap* gsoap)
	{
		// Delete instances
		soap_destroy(gsoap);
		// Delete data
		soap_end(gsoap);
		// Free soap struct engine context
		soap_free(gsoap);
	}
}