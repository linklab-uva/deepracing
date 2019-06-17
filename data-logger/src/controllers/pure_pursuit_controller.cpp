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
#include <Integrate.hpp>
#include "f1_datalogger/alglib/interpolation.h"
#include <unsupported/Eigen/Splines>
deepf1::PurePursuitController::PurePursuitController(std::shared_ptr<MeasurementHandler> measurement_handler,
	double Kv, double L, double max_angle, double velocity_setpoint)
{
	measurement_handler_ = measurement_handler;
	Kv_ = Kv;
	L_ = L;
	max_angle_ = max_angle;
	velocity_setpoint_ = velocity_setpoint;
	f1_interface_.reset(new deepf1::VJoyInterface);
}


deepf1::PurePursuitController::~PurePursuitController()
{
}
std::vector< std::pair< double, Eigen::Vector3d> > deepf1::PurePursuitController::loadTrackFile(const std::string& filename)
{
	std::vector< std::pair< double, Eigen::Vector3d> > rtn;
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
		rtn.push_back(std::make_pair(vec[0],p));
		//std::cout <<std::endl;
		//std::cout << eigenvec <<std::endl;
		//std::cout << std::endl;
	}
	std::vector< std::pair< double, Eigen::Vector3d > > interpolated_points;
	unsigned int max_index = rtn.size() - 3;
	double dt = (1.0/16.0);
	std::vector<double> T = { 0.0, (1.0/3.0), (2.0 / 3.0), 1.0 };
	alglib::real_1d_array t_alglib;
	t_alglib.attach_to_ptr(4, &T[0]);
	for (unsigned int i = 0; i < max_index; i++)
	{
		double x0 = rtn.at(i).first;
		double x1 = rtn.at(i + 1).first;
		double x2 = rtn.at(i + 2).first;
		double x3 = rtn.at(i+3).first;
		std::vector<double> X = { x0, x1, x2, x3};
		alglib::real_1d_array x_alglib;
		x_alglib.attach_to_ptr(4, &X[0]);
		alglib::spline1dinterpolant x_interpolant;
		alglib::spline1dbuildcubic(t_alglib, x_alglib, x_interpolant);
		double deltax = x3 - x0;
		Eigen::Vector3d P0 = rtn.at(i).second;
		Eigen::Vector3d P1 = rtn.at(i+1).second;
		Eigen::Vector3d P2 = rtn.at(i+2).second;
		Eigen::Vector3d P3 = rtn.at(i+3).second;
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
		[](const std::pair< double, Eigen::Vector3d>& a, const std::pair< double, Eigen::Vector3d>& b) 
		  { return a.first < b.first; });
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
void velocityControlLoop(std::shared_ptr<deepf1::MeasurementHandler> measurement_handler_,
	float velKp, float velKi, float velKd, float* setpoint, float* out)
{
	deepf1::TimestampedUDPData data;
	float speed, speed_mph;
	boost::circular_buffer<float> speed_buffer(10), time_buffer(10), error_buffer(10);
	_1D::TrapezoidRule<float> tr;
	while (true)
	{
		data = measurement_handler_->getData();

		speed = data.data.m_speed;
		speed_mph = speed * 2.23694;

		float current_error = *setpoint - speed_mph;
		error_buffer.push_back(current_error);
		speed_buffer.push_back(speed);
		time_buffer.push_back(data.data.m_time);
		if (speed_buffer.size() < 6 || time_buffer.size() < 6)
		{
			*out = 0.0;
			continue;
		}
		std::vector<float> errorvec(error_buffer.size());
		std::vector<float> timevec(time_buffer.size());
		for (unsigned int i = 0; i < error_buffer.size(); i++)
		{
			errorvec.push_back(error_buffer.at(i));
			timevec.push_back(time_buffer.at(i));
		}
		*out  = velKp * current_error + velKi * tr(timevec, errorvec) + velKd * (errorvec.back() - errorvec.at(errorvec.size() - 2)) / (timevec.back() - timevec.at(errorvec.size() - 2));
		std::this_thread::sleep_for(std::chrono::milliseconds(25));
	}
}
void deepf1::PurePursuitController::run(const std::string& trackfile, float velKp, float velKi, float velKd)
{
	deepf1::TimestampedUDPData data;
	F1ControlCommand commands;
	commands.throttle = 0.0;
	commands.steering = 0.0;
	commands.brake = 0.0;
	std::cout << "Setting initial command" << std::endl;
	f1_interface_->setCommands(commands);
	std::cout << "Set initial command" << std::endl;
	std::vector<std::pair<double,Eigen::Vector3d> > raceline = loadTrackFile(trackfile);
	int cols = raceline.size();
	Eigen::MatrixXd racelinematrix(3, cols);
	for (int i = 0; i < cols; ++i)
	{
		racelinematrix.col(i) = raceline[i].second;
	}
	kdt::KDTreed kdtree(racelinematrix);
	kdtree.build();
	kdt::KDTreed::Matrix dists;  
	kdt::KDTreed::MatrixI idx;
	float speed, speed_mph, lookahead_dist, fake_zero = 0.0, vel_setpoint, positive_deadband = fake_zero, negative_deadband = -fake_zero, accel;
	boost::circular_buffer<float> speed_buffer(10), time_buffer(10), error_buffer(10);
	_1D::TrapezoidRule<float> tr;
	vel_setpoint = velocity_setpoint_;
	std::thread control_loop(velocityControlLoop,measurement_handler_, velKp, velKi, velKd, &vel_setpoint, &accel);
	do
	{
		data = measurement_handler_->getData();
		speed = data.data.m_speed;
		lookahead_dist = std::max(6.0, Kv_ * (speed));
		Eigen::Vector3d forward(data.data.m_xd, data.data.m_yd, data.data.m_zd);
		Eigen::Vector3d right(data.data.m_xr, data.data.m_yr, data.data.m_zr);
		Eigen::Vector3d up = right.cross(forward);
		up.normalize();
		//std::printf("Position of Car: %f %f %f\n", data.data.m_x, data.data.m_y, data.data.m_z);
		Eigen::Vector3d position(data.data.m_x, data.data.m_y, data.data.m_z);
		Eigen::Vector3d real_axle_position = position;// -L_ * forward / 2;
		/*Eigen::MatrixXd queryPoint(3,1);
		queryPoint.col(0) = position;*/
		kdtree.query(real_axle_position, 1, idx, dists);
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
		//Eigen::MatrixXd lookaheadPoints = forwardPoints.leftCols((int)std::round(1.5*((double)min_index)));
		//std::pair<Eigen::Vector3d, Eigen::Vector3d> bfl = best_line_from_points(lookaheadPoints);
		//Eigen::ParametrizedLine< Eigen::Vector3d::Scalar, Eigen::Dynamic> line(bfl.first, bfl.second);
		//Eigen::Vector3d origin = lookaheadPoints.rowwise().mean();
		//double sstot = (lookaheadPoints.colwise() - origin).colwise().squaredNorm().sum(), ssres = 0;
		//for (unsigned int i = 0; i < lookaheadPoints.cols(); i++)
		//{
		//	ssres += line.squaredDistance(lookaheadPoints.col(i));
		//}
		//double R2 = 1.0 - (ssres / sstot);
		//std::printf("Forward Direction: %f %f %f\n", forward.x(), forward.y(), forward.z());
		//std::printf("Best fit line axis: %f %f %f\n", line.direction().x(), line.direction().y(), line.direction().z());
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
		double deadband = 0.025;
		if (delta < -1.0) delta = -1.0;
		else if (delta > 1.0) delta = 1.0;
		else if (std::abs(deadband) < deadband) delta = 0.0;
		commands.steering = delta;
		vel_setpoint = std::max(velocity_setpoint_ * std::pow(1- std::abs(alpha)/3.14, 7), 40.0);
		 
		if (accel < 0)
		{
			commands.brake = std::min(std::abs((double)accel),1.0);
			commands.throttle = 0.0;
		}
		else
		{
			commands.brake = 0.0;
			commands.throttle = std::min((double)accel, 1.0);
		}
		//std::printf("Suggested Steering %f. R2 value: %f\n", delta, R2);
		f1_interface_->setCommands(commands);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}while (true);
}