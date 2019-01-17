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
	unsigned int closestValueHelper(const std::vector<deepf1::protobuf::F1UDPData>& sorted_data,
	 int64_t search, unsigned int left, unsigned int right)
	{
		if(left == right)
		{
			return left;

		}
		else if((right-left)==1)
		{
			if(std::abs(sorted_data.at(left).logger_time() - search) < std::abs(sorted_data.at(right).logger_time() - search))
			{
				return left;
			}
			else
			{
				return right;
			}
		}
		else
		{
			unsigned int middle = (left + right)/2;
			int64_t query_val = sorted_data.at(middle).logger_time();
			if(query_val == search)
			{
				return middle;
			}
			else if(query_val < search)
			{
				return closestValueHelper(sorted_data, search, middle, right);
			}
			else
			{
				return closestValueHelper(sorted_data, search, left, middle);
			}
		}
	}
	std::pair<deepf1::protobuf::F1UDPData, unsigned int> PostProcessingUtils::closestValue(const std::vector<deepf1::protobuf::F1UDPData>& sorted_data, int64_t search)
	{
		unsigned int index = closestValueHelper(sorted_data, search, 0, sorted_data.size()-1);

		return std::pair<deepf1::protobuf::F1UDPData, unsigned int>(sorted_data.at(index), index);
	}
	std::vector<deepf1::protobuf::LabeledImage> PostProcessingUtils::labelImages(std::vector<deepf1::protobuf::F1UDPData>& udp_data, std::vector<deepf1::protobuf::TimestampedImage>& image_data, unsigned int interpolation_order)
	{
		printf("Labeling points.\n");
		std::vector<deepf1::protobuf::LabeledImage> rtn;
		std::sort(udp_data.begin(), udp_data.end(), &udpComp);
		std::sort(image_data.begin(), image_data.end(), &imageComp);

		rtn.resize(image_data.size());
		for(unsigned int i = 0; i < image_data.size(); i ++)
		{
			deepf1::protobuf::TimestampedImage image_point = image_data.at(i);	

			//printf("Processing image with filename: %s", image_point.image_file().c_str());
			
			std::pair<deepf1::protobuf::F1UDPData, unsigned int> pair = closestValue(udp_data, image_point.timestamp());
			deepf1::protobuf::F1UDPData closest_packet = udp_data.at(pair.second);

		//	printf("Image file %s with index #%u and timestamp %ld is closest to udp index %u with timestamp %ld. Delta=%ld\n", 
			//	image_point.image_file().c_str(), i, image_point.timestamp(), pair.second, closest_packet.logger_time(), closest_packet.logger_time()-image_point.timestamp());
			deepf1::protobuf::LabeledImage im;
			rtn.at(i).set_image_file(std::string(image_point.image_file()));
			rtn.at(i).set_brake(-1.0);
			rtn.at(i).set_throttle(-1.0);
			rtn.at(i).set_steering(closest_packet.steering());

			std::string json;
			google::protobuf::util::MessageToJsonString(rtn.at(i), &json);
			std::cout << "Labeled image: " << json << std::endl;
			//rtn.push_back(deepf1::protobuf::LabeledImage(im));
		}
	//	rtn.shrink_to_fit();

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
			bool success = data_in.ParseFromIstream(&stream_in);
			stream_in.close();
			if (!success)
			{

				std::cout << "FOUND EMPTY UDP PACKET" << std::endl;
				continue;
			}

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
			bool success = data_in.ParseFromIstream(&stream_in);
			stream_in.close();
			if (!success || data_in.image_file().empty())
			{

				std::cout << "FOUND EMPTY IMAGE FILENAME" << std::endl;
				continue;
			}
			/*
			std::string json;
			google::protobuf::util::MessageToJsonString(data_in, &json);
			std::cout << "Got data: " << std::endl << json << std::endl;
			*/
			rtn.push_back(data_in);
		}


		return rtn;

	}
}
}
