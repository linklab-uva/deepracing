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
	bool udpComp(const deepf1::protobuf::TimestampedUDPData& a, const deepf1::protobuf::TimestampedUDPData& b)
	{
		return a.timestamp() < b.timestamp();
	}
	bool imageComp(const deepf1::protobuf::TimestampedImage& a, const deepf1::protobuf::TimestampedImage& b)
	{
		return a.timestamp() < b.timestamp();
	}
	unsigned int closestValueHelper(const std::vector<deepf1::protobuf::TimestampedUDPData>& sorted_data,
	 google::protobuf::uint64 search, unsigned int left, unsigned int right)
	{
		if(left == right)
		{
			return left;

		}
		else if((right-left)==1)
		{
			if((search - sorted_data.at(left).timestamp()) < (sorted_data.at(right).timestamp() - search))
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
			google::protobuf::uint64 query_val = sorted_data.at(middle).timestamp();
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
	std::pair<deepf1::protobuf::TimestampedUDPData, unsigned int> PostProcessingUtils::closestValue(const std::vector<deepf1::protobuf::TimestampedUDPData>& sorted_data, google::protobuf::uint64 search)
	{
		unsigned int index = closestValueHelper(sorted_data, search, 0, sorted_data.size()-1);

		return std::pair<deepf1::protobuf::TimestampedUDPData, unsigned int>(sorted_data.at(index), index);
	}
	std::vector<float> interp(const std::vector<deepf1::protobuf::TimestampedUDPData>& udp_data, unsigned int closest_index, unsigned int interpolation_order, google::protobuf::uint64 image_timestamp)
	{
		std::vector<float> rtn(3);

		alglib::barycentricinterpolant steering_interp, brake_interp, throttle_interp;
		alglib::real_1d_array timestamps, steering, brake, throttle;
		timestamps.setlength(interpolation_order);
		steering.setlength(interpolation_order);
		brake.setlength(interpolation_order);
		throttle.setlength(interpolation_order);
		unsigned int lower_bound = closest_index - interpolation_order / 2;
		unsigned int upper_bound = closest_index + interpolation_order / 2 ;
		unsigned int idx = 0;
//		printf("Building alglib vectors at index %d. \n", closest_index);
		for (unsigned int i = lower_bound; i <= upper_bound; i++)
		{
			timestamps(idx) = (double)udp_data.at(i).timestamp();
			steering(idx) = (double)udp_data.at(i).udp_packet().m_steer();
			brake(idx) = (double)udp_data.at(i).udp_packet().m_brake();
			throttle(idx) = (double)udp_data.at(i).udp_packet().m_throttle();
			idx++;
		}
	//	printf("Built alglib vectors. \n");
		alglib::polynomialbuild(timestamps, steering, interpolation_order, steering_interp);
		alglib::polynomialbuild(timestamps, brake, interpolation_order, brake_interp);
		alglib::polynomialbuild(timestamps, throttle, interpolation_order, throttle_interp);
	//	printf("Built alglib models. \n");
		
		double t = (double)image_timestamp;
		rtn[0] = (float)alglib::barycentriccalc( steering_interp,  t);
		rtn[1] = (float)alglib::barycentriccalc( throttle_interp, t);
		rtn[2] = (float)alglib::barycentriccalc( brake_interp,  t);


		return rtn;
	}
	std::vector<deepf1::protobuf::LabeledImage> PostProcessingUtils::labelImages(std::vector<deepf1::protobuf::TimestampedUDPData>& udp_data, std::vector<deepf1::protobuf::TimestampedImage>& image_data, unsigned int interpolation_order)
	{
		printf("Labeling points.\n");
		std::vector<deepf1::protobuf::LabeledImage> rtn;
		std::sort(udp_data.begin(), udp_data.end(), &udpComp);
		std::sort(image_data.begin(), image_data.end(), &imageComp);

		rtn.reserve(image_data.size());
		for(unsigned int i = 0; i < image_data.size(); i ++)
		{
			deepf1::protobuf::TimestampedImage image_point = image_data.at(i);	

			//printf("Processing image with filename: %s", image_point.image_file().c_str());
			
			std::pair<deepf1::protobuf::TimestampedUDPData, unsigned int> pair = closestValue(udp_data, image_point.timestamp());
			deepf1::protobuf::TimestampedUDPData closest_packet = udp_data.at(pair.second);
			if ( pair.second < interpolation_order || pair.second >(udp_data.size() - interpolation_order) )
				continue; 

		//	printf("Image file %s with index #%u and timestamp %ld is closest to udp index %u with timestamp %ld. Delta=%ld\n", 
			//	image_point.image_file().c_str(), i, image_point.timestamp(), pair.second, closest_packet.logger_time(), closest_packet.logger_time()-image_point.timestamp());
			deepf1::protobuf::LabeledImage im;

		//	printf("Interpolating on order %u \n", interpolation_order);
			std::vector<float> interp_results = interp(udp_data, pair.second, interpolation_order, image_point.timestamp());
			im.mutable_label()->set_steering(interp_results[0]);
			im.mutable_label()->set_throttle(interp_results[1]);
			im.mutable_label()->set_brake(interp_results[2]);
			im.set_image_file(std::string(image_point.image_file()));
		//	printf("Done interpolating \n");
			/*
			std::string json;
			google::protobuf::util::MessageToJsonString(rtn.at(i), &json);
			std::cout << "Labeled image: " << json << std::endl;
			*/
			rtn.push_back(im);
		}
	//	rtn.shrink_to_fit();

		return rtn;
	}
	std::vector<deepf1::protobuf::TimestampedUDPData> PostProcessingUtils::parseUDPDirectory(const std::string& directory)
	{
		std::vector<deepf1::protobuf::TimestampedUDPData> rtn;
		
		std::vector<fs::path> paths;
		fs::path udp_dir(directory);
		get_all(udp_dir, ".pb", paths);
		std::ifstream stream_in;
		for (fs::path path : paths)
		{
			fs::path current_path = udp_dir / path;
	//		std::cout << "Loading file: " << current_path.string() << std::endl;
			stream_in.open(current_path.string().c_str());
			deepf1::protobuf::TimestampedUDPData data_in;
			bool success = data_in.ParseFromIstream(&stream_in);
			stream_in.close();
			if (!success || data_in.timestamp()==0)
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
		fs::path image_dir(directory);
		get_all(image_dir, ".pb", paths);
		std::ifstream stream_in;
		for (fs::path path : paths)
		{
			fs::path current_path = image_dir / path;
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
