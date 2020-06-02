#include "f1_datalogger/udp_logging/utils/eigen_utils.h"
#include <thread>
#include <iostream>
#include <google/protobuf/util/json_util.h>
#include <sstream>
#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"
#ifdef USE_ARMADILLO
#include <armadillo>
#endif
#include <stdexcept>
namespace deepf1
{ 

EigenUtils::EigenUtils()
{
}

EigenUtils::~EigenUtils()
{
}
Eigen::MatrixXd EigenUtils::loadArmaTxt(const std::string& armafile)
{
  #ifdef USE_ARMADILLO
  arma::Mat<double> arma_mat;
  if (!arma_mat.load(armafile, arma::arma_ascii))
  {
    throw std::runtime_error("Could not load arma txt file: " + armafile);
  }
  Eigen::MatrixXd rtn(Eigen::Map<Eigen::MatrixXd>(arma_mat.memptr(), arma_mat.n_rows, arma_mat.n_cols));
  return rtn;
  #else
  throw std::runtime_error("This feature only works with the armadillo library. Recompile with the WITH_ARMA option turned on");
  #endif
}
deepf1::protobuf::eigen::Pose3d EigenUtils::eigenToProto(const Eigen::Affine3d& poseEigen, const double& session_time, deepf1::protobuf::eigen::FrameId frameid)
{
  deepf1::protobuf::eigen::Pose3d rtn;

  Eigen::Vector3d translation(poseEigen.translation());
  Eigen::Quaterniond rotation(poseEigen.rotation());
  
  rtn.mutable_translation()->set_x(translation.x());
  rtn.mutable_translation()->set_y(translation.y());
  rtn.mutable_translation()->set_y(translation.z());

  rtn.mutable_rotation()->set_x(rotation.x());
  rtn.mutable_rotation()->set_y(rotation.y());
  rtn.mutable_rotation()->set_y(rotation.z());
  rtn.mutable_rotation()->set_w(rotation.w());

  rtn.set_frame(frameid);
  rtn.set_session_time(session_time);

  return rtn;
}
Eigen::Affine3d EigenUtils::protoToEigen(const deepf1::protobuf::eigen::Pose3d& poseProto)
{
  Eigen::Affine3d poseEigen;
  Eigen::Vector3d translation(poseProto.translation().x(), poseProto.translation().y(), poseProto.translation().z());
  Eigen::Quaterniond rotation(poseProto.rotation().w(), poseProto.rotation().x(), poseProto.rotation().y(), poseProto.rotation().z());
  poseEigen.fromPositionOrientationScale(translation, rotation, Eigen::Vector3d::Ones());
  return poseEigen;
}
Eigen::Affine3d EigenUtils::interpPoses(const Eigen::Affine3d& a, const Eigen::Affine3d& b, const double& s)
{
	Eigen::Affine3d rtn;
	Eigen::Vector3d translationA(a.translation());
	Eigen::Quaterniond rotationA(a.rotation());
	Eigen::Vector3d translationB(b.translation());
	Eigen::Quaterniond rotationB(b.rotation());

	Eigen::Vector3d translationOut = (1 - s) * translationA + s * translationB;
	Eigen::Quaterniond rotationOut = rotationA.slerp(s, rotationB);

	rtn.fromPositionOrientationScale(translationOut, rotationOut, Eigen::Vector3d::Ones());


	return rtn;
}
Eigen::Affine3d EigenUtils::motionPacketToPose(const deepf1::twenty_eighteen::CarMotionData& motion_packet)
{
	const deepf1::twenty_eighteen::protobuf::CarMotionData& motion_packet_pb =
		deepf1::twenty_eighteen::TwentyEighteenUDPStreamUtils::toProto(motion_packet);
	return motionPacketToPose(motion_packet_pb);
}
Eigen::Affine3d EigenUtils::motionPacketToPose(const deepf1::twenty_eighteen::protobuf::CarMotionData& motion_packet)
{
	Eigen::Affine3d rtn;
	Eigen::Vector3d translation(motion_packet.m_worldpositionx(), motion_packet.m_worldpositiony(), motion_packet.m_worldpositionz());

	Eigen::Vector3d forward(motion_packet.m_worldforwarddirx(), motion_packet.m_worldforwarddiry(), motion_packet.m_worldforwarddirz());
	forward.normalize();
	Eigen::Vector3d right(motion_packet.m_worldrightdirx(), motion_packet.m_worldrightdiry(), motion_packet.m_worldrightdirz());
	right.normalize();
	Eigen::Vector3d up = right.cross(forward);
	up.normalize();
	Eigen::Matrix3d rotationMat(Eigen::Matrix3d::Identity());
	rotationMat.col(0) = -right;
	rotationMat.col(1) = up;
	rotationMat.col(2) = forward;
	Eigen::Quaterniond rotation(rotationMat);

	rtn.fromPositionOrientationScale(translation, rotation, Eigen::Vector3d::Ones());

	return rtn;
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


}