#include "f1_datalogger/post_processing/post_processing_utils.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <google/protobuf/util/json_util.h>
#include "f1_datalogger/alglib/interpolation.h"
#include <sstream>
#include <Eigen/Geometry>
#include "f1_datalogger/controllers/kdtree_eigen.h"
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
	Eigen::MatrixXd PostProcessingUtils::readTrackFile(const std::string& trackfile)
	{
		Eigen::MatrixXd rtnMat;
		Eigen::Matrix<double,1,4> asdf = { 1.0, 2.0, 3.0, 4.0 };
		std::vector< std::pair< double, Eigen::Vector3d> > rtn;
		std::ifstream file;
		file.open(trackfile);
		std::string header;
		std::string labels;
		std::getline(file, header);
		std::cout << header << std::endl;
		std::getline(file, labels);
		std::cout << labels << std::endl;
		std::string line;

		while (true)
		{
			std::getline(file, line);
			if (file.eof())
			{
				break;
			}
			std::stringstream ss(line);
			std::vector<double> vec;
			while (ss.good())
			{
				std::string substr;
				std::getline(ss, substr, ',');
				vec.push_back(std::atof(substr.c_str()));
			}
			Eigen::Vector3d p(vec[1], vec[3], vec[2]);
			rtn.push_back(std::make_pair(vec[0], p));
			//std::cout <<std::endl;
			//std::cout << eigenvec <<std::endl;
			//std::cout << std::endl;
		}
		
		std::vector< std::pair< double, Eigen::Vector3d > > interpolated_points;
		unsigned int max_index = rtn.size() - 3;
		double dt = (1.0 / 16.0);
		std::vector<double> T = { 0.0, (1.0 / 3.0), (2.0 / 3.0), 1.0 };
		alglib::real_1d_array t_alglib;
		t_alglib.attach_to_ptr(4, &T[0]);
		for (unsigned int i = 0; i < max_index; i += 3)
		{
			double x0 = rtn.at(i).first;
			double x1 = rtn.at(i + 1).first;
			double x2 = rtn.at(i + 2).first;
			double x3 = rtn.at(i + 3).first;
			std::vector<double> X = { x0, x1, x2, x3 };
			alglib::real_1d_array x_alglib;
			x_alglib.attach_to_ptr(4, &X[0]);
			alglib::spline1dinterpolant x_interpolant;
			alglib::spline1dbuildcubic(t_alglib, x_alglib, x_interpolant);
			double deltax = x3 - x0;
			Eigen::Vector3d P0 = rtn.at(i).second;
			Eigen::Vector3d P1 = rtn.at(i + 1).second;
			Eigen::Vector3d P2 = rtn.at(i + 2).second;
			Eigen::Vector3d P3 = rtn.at(i + 3).second;
			for (double t = dt; t < 1.0; t += dt)
			{
				Eigen::Vector3d Pinterp = std::pow(1 - t, 3)*P0 + 3 * t*std::pow(1 - t, 2)*P1 +
					3 * std::pow(t, 2)*(1 - t)*P2 + std::pow(t, 3)*P3;
				double xinterp = alglib::spline1dcalc(x_interpolant, t);
				interpolated_points.push_back(std::make_pair(xinterp, Pinterp));
			}
		}
		rtn.insert(rtn.end(), interpolated_points.begin(), interpolated_points.end());
		std::sort(rtn.begin(), rtn.end(),
			[](const std::pair< double, Eigen::Vector3d>& a, const std::pair< double, Eigen::Vector3d >& b)
		{ return a.first < b.first; });
		rtnMat.resize(4, rtn.size());
		unsigned int idx = 0;

		std::for_each(rtn.begin(), rtn.end(), [&rtnMat, &idx](const std::pair< double, Eigen::Vector3d >& pair)
		{
			rtnMat(0, idx) = pair.first;
			rtnMat(Eigen::seqN(1, Eigen::last, 1), idx) = pair.second;
			idx++;
		});
		return rtnMat;
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
		get_all(udp_dir, ".json", paths);
		for (fs::path path : paths)
		{
			fs::path current_path = udp_dir / path;

			std::ifstream stream_in(current_path.string());
			std::stringstream buffer;
			buffer << stream_in.rdbuf();
			deepf1::protobuf::TimestampedUDPData data_in;
			google::protobuf::util::Status st = google::protobuf::util::JsonStringToMessage(buffer.str(), &data_in);
			bool success = st == google::protobuf::util::Status::OK;
			stream_in.close();
			if (!success || data_in.timestamp()==0)
			{

				std::cerr << "FOUND EMPTY UDP PACKET" << std::endl;
				continue;
			}
			else
			{
				//std::printf("Got a udp packet with Steering: %f, Throttle: %f, Brake: %f\n", data_in.udp_packet().m_steer(), data_in.udp_packet().m_throttle(), data_in.udp_packet().m_brake());
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
