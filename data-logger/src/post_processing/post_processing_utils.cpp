#include "f1_datalogger/post_processing/post_processing_utils.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <google/protobuf/util/json_util.h>
#include <alglib/interpolation.h>
namespace fs = boost::filesystem;
namespace deepf1
{
namespace post_processing
{
	PostProcessingUtils::PostProcessingUtils()
	{
	}
	PostProcessingUtils::~PostProcessingUtils()
	{
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
	bool udpComp(const deepf1::protobuf::F1UDPData& a, const deepf1::protobuf::F1UDPData& b)
	{
		return a.logger_time() < b.logger_time();
	}
	bool imageComp(const deepf1::protobuf::TimestampedImage& a, const deepf1::protobuf::TimestampedImage& b)
	{
		return a.timestamp() < b.timestamp();
	}
	deepf1::protobuf::F1UDPData PostProcessingUtils::closestValue(const std::vector<deepf1::protobuf::F1UDPData>& sorted_data, unsigned long long search)
	{
		const deepf1::protobuf::F1UDPData* raw_array = &(sorted_data[0]);
		deepf1::protobuf::F1UDPData rtn;
		unsigned int size = sorted_data.size();
		if (size == 1)
		{
			return sorted_data.at(0);
		}
		unsigned int middle_index = size / 2;
		deepf1::protobuf::F1UDPData middle_item = sorted_data.at(middle_index);
		if (middle_item.logger_time() == search)
		{
			return middle_item;
		}
		else if (search > middle_item.logger_time())
		{
			std::vector<deepf1::protobuf::F1UDPData>::const_iterator first = sorted_data.begin() + middle_index + 1;
			std::vector<deepf1::protobuf::F1UDPData>::const_iterator last = sorted_data.begin() + size - 1;
			std::vector<deepf1::protobuf::F1UDPData> newVec(first, last);
			return closestValue(newVec, search);
		}
		else
		{

		}





		return rtn;
	}
	std::vector<deepf1::protobuf::LabeledImage> PostProcessingUtils::labelImages(std::vector<deepf1::protobuf::F1UDPData>& udp_data, std::vector<deepf1::protobuf::TimestampedImage>& image_data, unsigned int interpolation_order)
	{
		std::vector<deepf1::protobuf::LabeledImage> rtn;
		std::sort(udp_data.begin(), udp_data.end(), &udpComp);
		std::sort(image_data.begin(), image_data.end(), &imageComp);


		return rtn;
	}
	std::vector<deepf1::protobuf::F1UDPData> PostProcessingUtils::parseUDPDirectory(const std::string& directory)
	{
		std::vector<deepf1::protobuf::F1UDPData> rtn;
		
		std::vector<fs::path> paths;
		fs::path udp_dir(directory);
		get_all(udp_dir, ".pb", paths);
		std::ifstream stream_in;
		for (fs::path path : paths)
		{
			fs::path current_path = udp_dir / path;
	//		std::cout << "Loading file: " << current_path.string() << std::endl;
			stream_in.open(current_path.string().c_str());
			deepf1::protobuf::F1UDPData data_in;
			data_in.ParseFromIstream(&stream_in);
			stream_in.close();
			//std::string json;
			//google::protobuf::util::MessageToJsonString(data_in, &json);
			//std::cout << "Got data: " << std::endl << json << std::endl;
			rtn.push_back(data_in);
		}
		

		return rtn;
	}
	std::vector<deepf1::protobuf::TimestampedImage> PostProcessingUtils::parseImageDirectory(const std::string& directory)
	{
		std::vector<deepf1::protobuf::TimestampedImage> rtn;

		std::vector<fs::path> paths;
		fs::path udp_dir(directory);
		get_all(udp_dir, ".pb", paths);
		std::ifstream stream_in;
		for (fs::path path : paths)
		{
			fs::path current_path = udp_dir / path;
			//		std::cout << "Loading file: " << current_path.string() << std::endl;
			stream_in.open(current_path.string().c_str());
			deepf1::protobuf::TimestampedImage data_in;
			data_in.ParseFromIstream(&stream_in);
			stream_in.close();
			//std::string json;
			//google::protobuf::util::MessageToJsonString(data_in, &json);
			//std::cout << "Got data: " << std::endl << json << std::endl;
			rtn.push_back(data_in);
		}


		return rtn;

	}
}
}
