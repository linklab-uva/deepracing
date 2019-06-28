#include "f1_datalogger/udp_logging/utils/eigen_utils.h"
#include <thread>
#include <boost/filesystem.hpp>
#include <iostream>
#include <google/protobuf/util/json_util.h>
#include "f1_datalogger/alglib/interpolation.h"
#include <sstream>
#include <Eigen/Geometry>
#include "f1_datalogger/controllers/kdtree_eigen.h"
namespace deepf1
{ 

EigenUtils::EigenUtils()
{
}

EigenUtils::~EigenUtils()
{
}
Eigen::MatrixXd EigenUtils::vectorToMatrix(const std::vector < Eigen::Vector4d >& vector)
{
	Eigen::MatrixXd rtnMat(4, vector.size());
	/**/
	rtnMat.resize(4, vector.size());
	unsigned int idx = 0;
	std::for_each(vector.begin(), vector.end(), [&rtnMat, &idx](const Eigen::Vector4d & point)
	{
		rtnMat.col(idx) = point;
		idx++;
	});
	return rtnMat;
}
std::vector < Eigen::Vector4d > EigenUtils::loadTrackFile(const std::string& trackfile, const double& interpolation_factor, bool debug)
{
	std::vector < Eigen::Vector4d > rtn;
	std::ifstream file;
	file.open(trackfile);
	std::string header;
	std::string labels;
	std::getline(file, header);
	if (debug)
	{
		std::cout << header << std::endl;
	}
	std::getline(file, labels);
	if (debug)
	{
		std::cout << labels << std::endl;
	}
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
		rtn.push_back(Eigen::Vector4d(vec[1], vec[3], vec[2], vec[0]));
	}
	if (interpolation_factor <= 0)
	{
		return rtn;
	}

	std::vector< Eigen::Vector4d > interpolated_points;
	unsigned int max_index = rtn.size() - 3;
	double dt = (1.0 / 16.0);
	std::vector<double> T = { 0.0, (1.0 / 3.0), (2.0 / 3.0), 1.0 };
	alglib::real_1d_array t_alglib;
	t_alglib.attach_to_ptr(4, &T[0]);
	for (unsigned int i = 0; i < max_index; i += 3)
	{
		double x0 = rtn.at(i).w();
		double x1 = rtn.at(i + 1).w();
		double x2 = rtn.at(i + 2).w();
		double x3 = rtn.at(i + 3).w();
		std::vector<double> X = { x0, x1, x2, x3 };
		alglib::real_1d_array x_alglib;
		x_alglib.attach_to_ptr(4, &X[0]);
		alglib::spline1dinterpolant x_interpolant;
		alglib::spline1dbuildcubic(t_alglib, x_alglib, x_interpolant);
		Eigen::Vector3d P0(rtn.at(i).x(), rtn.at(i).y(), rtn.at(i).z());
		Eigen::Vector3d P1(rtn.at(i + 1).x(), rtn.at(i + 1).y(), rtn.at(i + 1).z());
		Eigen::Vector3d P2(rtn.at(i + 2).x(), rtn.at(i + 2).y(), rtn.at(i + 2).z());
		Eigen::Vector3d P3(rtn.at(i + 3).x(), rtn.at(i + 3).y(), rtn.at(i + 3).z());
		for (double t = interpolation_factor; t < 1.0; t += interpolation_factor)
		{
			double x = alglib::spline1dcalc(x_interpolant, t);
			Eigen::Vector3d Pinterp = std::pow(1 - t, 3)*P0 + 3 * t*std::pow(1 - t, 2)*P1 +
				3 * std::pow(t, 2)*(1 - t)*P2 + std::pow(t, 3)*P3;
			interpolated_points.push_back( Eigen::Vector4d( Pinterp.x(), Pinterp.y(), Pinterp.z(), x) );
		}
	}
	rtn.insert(rtn.end(), interpolated_points.begin(), interpolated_points.end());
	std::sort(rtn.begin(), rtn.end(),
		[](const Eigen::Vector4d& a, const Eigen::Vector4d& b)
	{ return a.w() < b.w(); });
	/*rtnMat.resize(4, rtn.size());
	unsigned int idx = 0;

	std::for_each(rtn.begin(), rtn.end(), [&rtnMat, &idx](const std::pair< double, Eigen::Vector3d >& pair)
	{
		rtnMat(0, idx) = pair.first;
		rtnMat(Eigen::seqN(1, Eigen::last, 1), idx) = pair.second;
		idx++;
	});
	return rtnMat;*/
	return rtn;
}

}