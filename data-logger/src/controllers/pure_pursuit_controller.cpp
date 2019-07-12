#include "f1_datalogger/controllers/pure_pursuit_controller.h"
#include <thread>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Dense>
#include "f1_datalogger/controllers/kdtree_eigen.h"
#include "f1_datalogger/alglib/interpolation.h"
#include "f1_datalogger/controllers/f1_interface_factory.h"
#include <boost/circular_buffer.hpp>
#include "f1_datalogger/udp_logging/utils/eigen_utils.h"
deepf1::PurePursuitController::PurePursuitController(std::shared_ptr<MeasurementHandler2018> measurement_handler,
	double Kv, double L, double max_angle, double velocity_setpoint)
{
	measurement_handler_ = measurement_handler;
	Kv_ = Kv;
	L_ = L;
	max_angle_ = max_angle;
	velocity_setpoint_ = velocity_setpoint;
  f1_interface_ = deepf1::F1InterfaceFactory::getDefaultInterface();
}


deepf1::PurePursuitController::~PurePursuitController()
{
}
void velocityControlLoop(std::shared_ptr<deepf1::MeasurementHandler2018> measurement_handler_,
	float velKp, float velKi, float velKd, float* setpoint, float* out)
{
	deepf1::twenty_eighteen::PacketCarTelemetryData telemetry_data;
	deepf1::twenty_eighteen::PacketMotionData motion_Data;
	float speed, speed_mph;
	
	boost::circular_buffer<float> speed_buffer(10), time_buffer(10), error_buffer(10);
	while (true)
	{
	//	telemetry_data = deepf1::twenty_eighteen::PacketCarTelemetryData(measurement_handler_->getCurrentTelemetryData().data);
		motion_Data = deepf1::twenty_eighteen::PacketMotionData(measurement_handler_->getCurrentMotionData().data);
		Eigen::Vector3d velocity_vector(motion_Data.m_carMotionData[0].m_worldVelocityX,
			motion_Data.m_carMotionData[0].m_worldVelocityY, motion_Data.m_carMotionData[0].m_worldVelocityZ);
		speed = velocity_vector.norm();
		speed_mph = speed * 2.23694;

		float current_error = *setpoint - speed_mph;
		//std::printf("Current speed: %f. Current setpoint: %f. Current error: %f\n", speed_mph, *setpoint, current_error);
		if (isnan(current_error))
		{
			//*out = 1.0;
			continue;
		}
		error_buffer.push_back(current_error);
		speed_buffer.push_back(speed);
		time_buffer.push_back(motion_Data.m_header.m_sessionTime);
		if (speed_buffer.size() < 6 || time_buffer.size() < 6)
		{
			*out = 0.0;
			continue;
		}

		float* errorptr = error_buffer.linearize();
		std::vector<float> errorvec(errorptr, errorptr + error_buffer.size());

		float* timeptr = time_buffer.linearize();
		std::vector<float> timevec(timeptr, timeptr + time_buffer.size());

		float integral = 0.0;
		unsigned int upperbound = timevec.size() - 1;
		for (unsigned int i = 0; i < upperbound; i++)
		{
			if( isnan(errorvec[i]) || isnan(errorvec[i+1]) || isnan(timevec[i + 1]) || isnan(timevec[i]) )
			{
				continue;
			}
			integral += 0.5* (errorvec[i] + errorvec[i + 1])*(timevec[i + 1] - timevec[i]);
		}
		float derivative = (errorvec.back() - errorvec.at(errorvec.size() - 2)) / (timevec.back() - timevec.at(errorvec.size() - 2));
		if (isnan(derivative))
		{
			derivative = 0.0;
		}
		if(isnan(integral))
		{
			integral = 0.0;
		}
		*out  = velKp * current_error + velKi * integral + velKd * derivative;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}
void deepf1::PurePursuitController::run(const std::string& trackfile, float velKp, float velKi, float velKd, float velocity_lookahead_gain)
{
	F1ControlCommand commands;
	commands.throttle = 0.0;
	commands.steering = 0.0;
	commands.brake = 0.0;
	std::cout << "Setting initial command" << std::endl;
	f1_interface_->setCommands(commands);
	std::cout << "Set initial command" << std::endl;
	//std::vector<std::pair<double,Eigen::Vector3d> > raceline = loadTrackFile(trackfile);
//	std::vector<Eigen::Vector4d> raceline = deepf1::EigenUtils::loadTrackFile(trackfile, 1.0/16.0);
 // Eigen::MatrixXd racelinematrixFull = deepf1::EigenUtils::vectorToMatrix(raceline);
  Eigen::MatrixXd racelinematrixFull = deepf1::EigenUtils::loadArmaTxt(trackfile);
  racelinematrixFull.transposeInPlace();
	int cols = racelinematrixFull.cols();
	Eigen::MatrixXd racelinematrix = racelinematrixFull(Eigen::seqN(1,3), Eigen::all);
  Eigen::MatrixXd racelinematrixAugmented = Eigen::MatrixXd::Ones(4, racelinematrix.cols());
  racelinematrixAugmented(Eigen::seqN(0, 3), Eigen::all) = racelinematrix;
  Eigen::MatrixXd velocitymatrix = racelinematrixFull(Eigen::seqN(4, 3), Eigen::all);
	kdt::KDTreed kdtree(racelinematrix);
	kdtree.build();
	kdt::KDTreed::Matrix dists;  
	kdt::KDTreed::MatrixI idx;
  float speed, speed_mph, velocity_lookahead_dist, lookahead_dist, fake_zero = 0.0, vel_setpoint, positive_deadband = fake_zero, negative_deadband = -fake_zero, accel;
	boost::circular_buffer<float> speed_buffer(10), time_buffer(10), error_buffer(10);
	vel_setpoint = velocity_setpoint_;
	std::thread control_loop(velocityControlLoop,measurement_handler_, velKp, velKi, velKd, &vel_setpoint, &accel);
	deepf1::twenty_eighteen::PacketCarTelemetryData telemetry_data;
	deepf1::twenty_eighteen::PacketMotionData motion_Data;
	do
	{
	//	telemetry_data = deepf1::twenty_eighteen::PacketCarTelemetryData(measurement_handler_->getCurrentTelemetryData().data);
		motion_Data = deepf1::twenty_eighteen::PacketMotionData(measurement_handler_->getCurrentMotionData().data);
		Eigen::Vector3d velocity_vector(motion_Data.m_carMotionData[0].m_worldVelocityX,
			motion_Data.m_carMotionData[0].m_worldVelocityY, motion_Data.m_carMotionData[0].m_worldVelocityZ);
		speed = velocity_vector.norm();
		lookahead_dist = std::max(6.0, Kv_ * (speed));
		velocity_lookahead_dist = (float) std::max(6.0,  ((double)velocity_lookahead_gain * (double)(speed)));
		Eigen::Vector3d forward(motion_Data.m_carMotionData[0].m_worldForwardDirX, motion_Data.m_carMotionData[0].m_worldForwardDirY, motion_Data.m_carMotionData[0].m_worldForwardDirZ);
		forward = (1.0 / 32767.0)*forward;
		forward.normalize();
		Eigen::Vector3d right(motion_Data.m_carMotionData[0].m_worldRightDirX, motion_Data.m_carMotionData[0].m_worldRightDirY, motion_Data.m_carMotionData[0].m_worldRightDirZ);
		right = (1.0 / 32767.0)*right;
		right.normalize();
		Eigen::Vector3d up = right.cross(forward);
		up.normalize();
		Eigen::Vector3d position(motion_Data.m_carMotionData[0].m_worldPositionX, motion_Data.m_carMotionData[0].m_worldPositionY, motion_Data.m_carMotionData[0].m_worldPositionZ);
	//	Eigen::Vector3d real_axle_position = position;// -L_ * forward / 2;
		//std::printf("Position of Car: %f %f %f\n", position.x(), position.y(), position.z());
		//std::printf("Forward Vector: %f %f %f\n", forward.x(), forward.y(), forward.z());
		/*Eigen::MatrixXd queryPoint(3,1);
		queryPoint.col(0) = position;*/
    Eigen::Matrix3d currentRotMatrix = Eigen::Matrix3d::Identity();
    currentRotMatrix.col(0) = -right;
    currentRotMatrix.col(1) = up;
    currentRotMatrix.col(2) = forward;
    Eigen::Quaterniond currentQuat(currentRotMatrix);
    Eigen::Affine3d currentPose;
    currentPose.fromPositionOrientationScale(position, currentQuat, Eigen::Vector3d::Ones());
    Eigen::MatrixXd pointsLocal = ((currentPose.inverse().matrix()) * racelinematrixAugmented)(Eigen::seqN(0, 3), Eigen::all);
    Eigen::MatrixXd velocitiesLocal = ((currentPose.rotation().inverse().matrix()) * velocitymatrix);
    Eigen::VectorXd localNorms = pointsLocal.colwise().norm();
    Eigen::VectorXd::Index min_index;
    localNorms.minCoeff(&min_index);

		//kdtree.query(real_axle_position, 1, idx, dists);
  //  unsigned int raceline_index = idx(0);
    unsigned int raceline_index = min_index;
		//Eigen::Vector3d closestpoint(racelinematrix.col(raceline_index));
		//std::printf("Closest point in raceline: %f %f %f\n", closestpoint.x(), closestpoint.y(), closestpoint.z());
		unsigned int num_cols = pointsLocal.cols() - raceline_index;
		Eigen::MatrixXd forwardPoints(pointsLocal.rightCols(num_cols));
    //std::cout << std::endl;
    //std::cout << forwardPoints(Eigen::all,  Eigen::seqN(0, 6) ) << std::endl;
    //std::cout << std::endl;
    Eigen::MatrixXd forwardVelocities(velocitiesLocal.rightCols(num_cols));
		Eigen::VectorXd forwardDiffs = forwardPoints.colwise().norm();
	//	std::cout << "Forward Diffs has size : " << forwardDiffs.size() << std::endl;
		Eigen::VectorXd lookaheadDiffs = (forwardDiffs - Eigen::VectorXd::Ones(num_cols)*lookahead_dist).cwiseAbs();
		Eigen::VectorXd lookaheadDiffsVelocity = (forwardDiffs - Eigen::VectorXd::Ones(num_cols)*velocity_lookahead_dist).cwiseAbs();

		//std::cout << "Lookahead Diffs has size : " << lookaheadDiffs.size() << std::endl;
    lookaheadDiffs.minCoeff(&min_index);

		Eigen::VectorXd::Index min_index_velocity;
		lookaheadDiffsVelocity.minCoeff(&min_index_velocity);

		Eigen::Vector3d lookaheadPoint = forwardPoints.col(min_index);
		Eigen::Vector3d lookaheadPointVelocity = forwardPoints.col(min_index_velocity);
    Eigen::Vector3d lookaheadVelocity = forwardVelocities.col(min_index_velocity);
		//Eigen::Vector3d lookaheadStart = forwardPoints.col(0);
		//if ((lookaheadPoint - lookaheadStart).norm() < 1.0)
		//{
		//	lookaheadPoint = forwardPoints.col(2);
		//}
		//if ((lookaheadPointVelocity - lookaheadStart).norm() < 1.0)
		//{
		//	lookaheadPointVelocity = forwardPoints.col(2);
		//}
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
    Eigen::Vector3d lookaheadVector = (lookaheadPoint);// -real_axle_position);
    lookaheadVector.normalize();
    Eigen::Vector3d lookaheadVectorVelocity = (lookaheadPointVelocity);// -real_axle_position);
		lookaheadVectorVelocity.normalize();
		//Eigen::Vector3d crossVector = forward.cross(lookaheadVector);
		//crossVector.normalize();
		//std::printf("Cross Vector: %f %f %f\n", crossVector.x(), crossVector.y(), crossVector.z());
		double alpha = std::abs(std::acos(lookaheadVector.dot(Eigen::Vector3d::UnitZ())));
    lookaheadVelocity.y() = 0;
    lookaheadVelocity.normalize();
		double alphaVelocity = std::abs(std::acos(lookaheadVelocity.dot(Eigen::Vector3d::UnitZ())));
		if (lookaheadVector.x() < 0)
		{
			alpha *= -1.0;
			alphaVelocity *= -1.0;
		}
		double physical_angle = -1.0* std::atan((2 * L_*std::sin(alpha)) / lookahead_dist);
		double delta;
		if (physical_angle > 0)
		{
			delta = -3.79616039*physical_angle + 0.01004506;
		}
		else
		{
			delta = -3.34446413*physical_angle + 0.01094534;
		}
		double deadband = (1.0/64.0);
		if (delta < -1.0) delta = -1.0;
		else if (delta > 1.0) delta = 1.0;
		else if (std::abs(deadband) < deadband) delta = 0.0;
		commands.steering = delta;
    double ratio_complement = std::abs(lookaheadVelocity.z());
		double vel_factor;
		if (ratio_complement>.850)
		{
			vel_factor = 1.0;
		}
		else if (ratio_complement > .65)
		{
			vel_factor = std::pow(ratio_complement, 1);
		}
		else
		{
			vel_factor = std::pow(ratio_complement, 2);
		}
		vel_setpoint = std::max(velocity_setpoint_ * vel_factor, 50.0);
		 
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