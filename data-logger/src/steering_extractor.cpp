#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "math_utils.h" 
#include "deepf1_gsoap/deepf1_gsoapH.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace deepf1 {

	void cleanup_soap(soap* gsoap);
	void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret);
	bool sort_udp_functor(deepf1_gsoap::ground_truth_sample * a, deepf1_gsoap::ground_truth_sample * b);
}
int main(int argc, char** argv) {

	unsigned int interpolation_degree;
	std::string data_directory, annotation_prefix, output_file;
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "Displays options and exits")
		("output_folder,f", po::value<std::string>(&data_directory)->default_value(std::string("data")), "Top-level folder to look for the original data dump.")
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



	std::ofstream file_out(output_file, std::fstream::out);
	soap* file_reader = soap_new();
	
	printf("Loading raw annotations from file.\n");
	fs::path annotations_dir = root_dir / fs::path("raw_annotations");

	if (!fs::is_directory(annotations_dir)) {
		std::cerr << "ERROR: Annotations directory " << root_dir.string() << "does not exist." << std::endl;
		exit(-1);
	}

	std::vector<fs::path> filez;
	deepf1::get_all(annotations_dir, std::string(".xml"), filez);
	std::vector<::deepf1_gsoap::ground_truth_sample*> samples;
	std::shared_ptr<std::fstream> file_in;
	for (unsigned int i = 0; i < filez.size(); i++) {
		fs::path current_file = annotations_dir / filez.at(i);
	//	std::cout << "Processing file: " << current_file.string() << std::endl;
		file_in.reset(new std::fstream(current_file.string(), std::fstream::in));
		samples.push_back(deepf1_gsoap::soap_new_ground_truth_sample(file_reader));
		file_reader->is = file_in.get();
		deepf1_gsoap::soap_read_ground_truth_sample(file_reader, samples.back());
	}
	printf("Got the files. Sorting by timestamp.\n");
	std::sort(samples.begin(), samples.end(),&deepf1::sort_udp_functor);
	printf("Done sorting, doing polynomial interpolation.\n");
	std::vector<double> udp_x, udp_y;
	udp_x.reserve(samples.size());
	udp_y.reserve(samples.size());
	for (auto it = samples.begin(); it != samples.end(); it++) {
		udp_x.push_back((double)(*it)->timestamp);
		udp_y.push_back((double)(*it)->sample.m_steer);
	}
	fs::path images_dir = root_dir / fs::path("raw_images");
	fs::path image_timestamps = images_dir / fs::path("raw_image_timestamps.csv");
	std::ifstream image_names_in(image_timestamps.string().c_str(), std::fstream::in);
	while (!image_names_in.eof()) {
		std::string image, ts_string;
		getline(image_names_in, image,',');
		getline(image_names_in, ts_string);
		double ts = std::atof(ts_string.c_str());
		std::vector<double>::iterator to_comp = ::std::lower_bound(udp_x.begin(), udp_x.end(), ts);
		std::vector<double>::iterator closest;
		if (to_comp == udp_x.begin())
		{
			closest = udp_x.begin();
		}
		else if (to_comp == udp_x.end())
		{
			closest = udp_x.end();
		}
		else if (std::abs(ts - *(to_comp-1)) < std::abs(ts - *(to_comp))) {
			closest = to_comp-1;
		}
		else {
			closest = to_comp;
		}
		unsigned int index = closest - udp_x.begin();
		if (index <= interpolation_degree || (udp_x.size() - index) <= interpolation_degree + 1) {
			continue;
		}

		alglib::real_1d_array alglib_udp_x, alglib_udp_y;
		unsigned int start_index = index - std::ceil((double)interpolation_degree / 2.0);
		alglib_udp_x.setcontent(interpolation_degree + 1, &udp_x[start_index]);
		alglib_udp_y.setcontent(interpolation_degree + 1, &udp_y[start_index]);
		alglib::barycentricinterpolant p;
		alglib::polynomialbuild(alglib_udp_x,alglib_udp_y,p);
		double interp, interp_d;
		alglib::barycentricdiff1(p, ts, interp, interp_d);
		file_out << image << "," << ts << ","<< interp << std::endl;	
	}
	deepf1::cleanup_soap(file_reader);
	return 0;
}
namespace deepf1 {
	bool sort_udp_functor(deepf1_gsoap::ground_truth_sample * a, deepf1_gsoap::ground_truth_sample * b) {
		return a->timestamp < b->timestamp;
	}
	void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret)
	{
		if (!fs::exists(root) || !fs::is_directory(root)) return;

		fs::recursive_directory_iterator it(root);
		fs::recursive_directory_iterator endit;

		while (it != endit)
		{
			if (fs::is_regular_file(*it) && it->path().extension() == ext) ret.push_back(it->path().filename());
			++it;

		}

	}
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