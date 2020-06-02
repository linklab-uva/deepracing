#ifndef F1_DATALOGGER_EIGEN_UTILS_H
#define F1_DATALOGGER_EIGEN_UTILS_H
#include <vector>
#include <Eigen/Geometry>
#include "f1_datalogger/proto/CarMotionData.pb.h"
#include "f1_datalogger/car_data/car_data.h"
#include "f1_datalogger/proto/Pose3d.pb.h"
#include <f1_datalogger/visibility_control.h>


namespace deepf1
{
	class F1_DATALOGGER_PUBLIC EigenUtils
	{
	public:
		EigenUtils();
		~EigenUtils();
    static Eigen::MatrixXd loadArmaTxt(const std::string& armafile);
	static Eigen::MatrixXd vectorToMatrix(const std::vector < Eigen::Vector4d >& vector);
	static Eigen::Affine3d motionPacketToPose(const deepf1::twenty_eighteen::protobuf::CarMotionData& motion_packet);
	static Eigen::Affine3d motionPacketToPose(const deepf1::twenty_eighteen::CarMotionData& motion_packet);
	static Eigen::Affine3d interpPoses(const Eigen::Affine3d& a, const Eigen::Affine3d& b, const double& s);
    static Eigen::Affine3d protoToEigen(const deepf1::protobuf::eigen::Pose3d& poseProto);
    static deepf1::protobuf::eigen::Pose3d eigenToProto(const Eigen::Affine3d& poseEigen, const double& session_time, deepf1::protobuf::eigen::FrameId frameid);
	};

}
#endif