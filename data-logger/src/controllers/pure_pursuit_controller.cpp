#include "f1_datalogger/controllers/pure_pursuit_controller.h"
#include <thread>
#include <vJoy++/vjoy.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include "f1_datalogger/controllers/kdtree_eigen.h"
deepf1::PurePursuitController::PurePursuitController(std::shared_ptr<MeasurementHandler> measurement_handler,
	double Kv, double L, double max_angle, double throttle)
{
	measurement_handler_ = measurement_handler;
	Kv_ = Kv;
	L_ = L;
	max_angle_ = max_angle;
	throttle_ = throttle;
}


deepf1::PurePursuitController::~PurePursuitController()
{
}

std::vector<Eigen::Vector3d> deepf1::PurePursuitController::loadTrackFile(const std::string& filename)
{
	std::vector<Eigen::Vector3d> rtn;
	std::ifstream file;
	file.open(filename);
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
		rtn.push_back(p);
		//std::cout <<std::endl;
		//std::cout << eigenvec <<std::endl;
		//std::cout << std::endl;
	}
	

	return rtn;
}

void deepf1::PurePursuitController::run(const std::string& trackfile)
{
	deepf1::TimestampedUDPData data;
	vjoy_plusplus::vJoy vjoy(1);
	vjoy_plusplus::JoystickPosition js;
	double max_vjoysteer = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoythrottle = (double)vjoy_plusplus::vJoy::maxAxisvalue(), max_vjoybrake = (double)vjoy_plusplus::vJoy::maxAxisvalue();
	double middle_vjoysteer = max_vjoysteer / 2.0;
	js.lButtons = 0x00000000;
	js.wAxisX = 0;
	js.wAxisY = 0;
	js.wAxisXRot = std::round(max_vjoythrottle*throttle_);
	js.wAxisYRot = 0;
	vjoy.update(js);
	std::vector<Eigen::Vector3d> raceline = loadTrackFile(trackfile);
	int cols = raceline.size();
	Eigen::MatrixXd racelinematrix(3, cols);
	for (int i = 0; i < cols; ++i)
	{
		racelinematrix.col(i) = raceline[i];
	}
	kdt::KDTreed kdtree(racelinematrix);
	kdtree.build();
	float speed, lookahead_dist;
	kdt::KDTreed::Matrix dists;  
	kdt::KDTreed::MatrixI idx;
	float fake_zero = 0.0;
	float positive_deadband = fake_zero, negative_deadband = -fake_zero;
	do
	{
		data = measurement_handler_->getData();
		speed = data.data.m_speed;
		lookahead_dist = std::max(Kv_ * speed,0.1);
		Eigen::Vector3d forward(data.data.m_xd, data.data.m_yd, data.data.m_zd);
		Eigen::Vector3d right(data.data.m_xr, data.data.m_yr, data.data.m_zr);
		Eigen::Vector3d up = right.cross(forward);
		up.normalize();
	//	std::printf("Forward Direction: %f %f %f\n", forward.x(), forward.y(), forward.z());
	//	std::printf("Up Direction: %f %f %f\n", up.x(), up.y(), up.z());
		Eigen::Vector3d position(data.data.m_x, data.data.m_y, data.data.m_z);
		Eigen::Vector3d real_axle_position = position - L_*forward;
		/*Eigen::MatrixXd queryPoint(3,1);
		queryPoint.col(0) = position;*/
		kdtree.query(real_axle_position, 1, idx, dists);
		std::printf("Cars XYZ real axle position: %f %f %f\n", real_axle_position.x(), real_axle_position.y(), real_axle_position.z());
		unsigned int raceline_index = idx(0);
		Eigen::Vector3d closestpoint(racelinematrix.col(raceline_index));
		//std::printf("Closest point in raceline: %f %f %f\n", closestpoint.x(), closestpoint.y(), closestpoint.z());
		unsigned int num_cols = racelinematrix.cols() - raceline_index;
		Eigen::MatrixXd forwardPoints(racelinematrix.rightCols(num_cols));
		Eigen::VectorXd forwardDiffs = (forwardPoints.colwise() - real_axle_position).colwise().norm();
	//	std::cout << "Forward Diffs has size : " << forwardDiffs.size() << std::endl;
		Eigen::VectorXd lookaheadDiffs = (forwardDiffs - lookahead_dist*Eigen::VectorXd::Ones(forwardDiffs.size())).cwiseAbs();
		//std::cout << "Lookahead Diffs has size : " << lookaheadDiffs.size() << std::endl;
		Eigen::VectorXd::Index min_index;
		lookaheadDiffs.minCoeff(&min_index);

		Eigen::Vector3d lookaheadPoint = forwardPoints.col(min_index);
		std::printf("Lookahead point: %f %f %f\n", lookaheadPoint.x(), lookaheadPoint.y(), lookaheadPoint.z());
		Eigen::Vector3d lookaheadVector = (lookaheadPoint - real_axle_position);
		lookaheadVector.normalize();
		Eigen::Vector3d crossVector = forward.cross(lookaheadVector);
		crossVector.normalize();
		//std::printf("Cross Vector: %f %f %f\n", crossVector.x(), crossVector.y(), crossVector.z());
		double alpha = std::abs(std::acos(lookaheadVector.dot(forward)));
		if (crossVector.y() < 0)
		{
			alpha *= -1.0;
		}
		double delta =  std::atan((2 * L_*std::sin(alpha)) / lookahead_dist)  / max_angle_;
		std::printf("Suggested Steering %f\n", delta);
		if (delta > 1.0)
		{
			delta = 1.0;
		}
		else if (delta < -1.0)
		{
			delta = -1.0;
		}
		if (delta > positive_deadband)
		{
			js.wAxisX = std::round(max_vjoysteer*alpha);
			js.wAxisY = 0;
		}
		else if (delta < negative_deadband)
		{
			js.wAxisX = 0;
			js.wAxisY = std::round(max_vjoysteer * std::abs(alpha));
		}
		else
		{
			js.wAxisX = 0;
			js.wAxisY = 0;
		}
		js.wAxisXRot = std::round(max_vjoythrottle*throttle_);
		js.wAxisYRot = 0;
		vjoy.update(js);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}while (true);
}