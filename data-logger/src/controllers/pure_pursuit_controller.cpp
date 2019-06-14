#include "f1_datalogger/controllers/pure_pursuit_controller.h"
#include <thread>
#include "f1_datalogger/controllers/vjoy_interface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include "f1_datalogger/controllers/kdtree_eigen.h"
#include <boost/circular_buffer.hpp>
deepf1::PurePursuitController::PurePursuitController(std::shared_ptr<MeasurementHandler> measurement_handler,
	double Kv, double L, double max_angle, double throttle)
{
	measurement_handler_ = measurement_handler;
	Kv_ = Kv;
	L_ = L;
	max_angle_ = max_angle;
	throttle_ = throttle;
	f1_interface_.reset(new deepf1::VJoyInterface);
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
std::pair < Eigen::Vector3d, Eigen::Vector3d > best_line_from_points(const Eigen::MatrixXd & points)
{
	// copy coordinates to  matrix in Eigen format
	Eigen::MatrixXd centers = points.transpose();
	

	Eigen::Vector3d origin = centers.colwise().mean();
	Eigen::MatrixXd centered = centers.rowwise() - origin.transpose();
	Eigen::MatrixXd cov = centered.adjoint() * centered;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
	Eigen::Vector3d axis = eig.eigenvectors().col(2).normalized();

	return std::make_pair(origin, axis);
}
void deepf1::PurePursuitController::run(const std::string& trackfile)
{
	deepf1::TimestampedUDPData data;
	std::cout << "Making vjoy interface" << std::endl;
	std::cout << "Made vjoy interface" << std::endl;
	F1ControlCommand commands;
	commands.throttle = throttle_;
	std::cout << "Setting initial command" << std::endl;
	f1_interface_->setCommands(commands);
	std::cout << "Set initial command" << std::endl;
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
		speed = data.data.m_speed * 0.277778;
		lookahead_dist = std::min(6.0, Kv_ * (speed));
		Eigen::Vector3d forward(data.data.m_xd, data.data.m_yd, data.data.m_zd);
		Eigen::Vector3d right(data.data.m_xr, data.data.m_yr, data.data.m_zr);
		Eigen::Vector3d up = right.cross(forward);
		up.normalize();
	//	std::printf("Forward Direction: %f %f %f\n", forward.x(), forward.y(), forward.z());
	//	std::printf("Up Direction: %f %f %f\n", up.x(), up.y(), up.z());
		Eigen::Vector3d position(data.data.m_x, data.data.m_y, data.data.m_z);
		Eigen::Vector3d real_axle_position = position - L_*forward/2;
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
		Eigen::Vector3d lookaheadStart = forwardPoints.col(0);
		if ((lookaheadPoint - lookaheadStart).norm() < 1.0)
		{
			lookaheadPoint = forwardPoints.col(2);
		}
		//Eigen::MatrixXd lookaheadPoints = forwardPoints.leftCols(50);
		//std::pair<Eigen::Vector3d, Eigen::Vector3d> bfl = best_line_from_points(lookaheadPoints);
		//Eigen::ParametrizedLine< Eigen::Vector3d::Scalar, Eigen::Dynamic> line(bfl.first, bfl.second);
		//Eigen::Vector3d origin = lookaheadPoints.rowwise().mean();
		//double sstot = (lookaheadPoints.colwise() - origin).colwise().squaredNorm().sum(), ssres = 0;
		//for (unsigned int i = 0; i < lookaheadPoints.cols(); i++)
		//{
		//	ssres += line.squaredDistance(lookaheadPoints.col(i));
		//}
		//double R2 = 1.0 - (ssres / sstot);
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
		commands.steering = delta;
		double accel = throttle_;
		
		//if (accel < 0)
		//{
		//	commands.brake = std::min(std::abs(accel),1.0);
		//	commands.throttle = 0.0;
		//}
		//else
		//{
		//	commands.brake = 0.0;
		//	commands.throttle = std::min(accel, 1.0);
		//}
		commands.brake = 0.0;
		commands.throttle = throttle_;
		std::printf("Resolved lookahead distance: %f. accel: %f\n", (lookaheadPoint - lookaheadStart).norm(), accel);
		std::printf("Suggested Steering %f\n", delta);
		f1_interface_->setCommands(commands);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}while (true);
}