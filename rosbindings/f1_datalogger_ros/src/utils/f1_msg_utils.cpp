#include "f1_datalogger_ros/utils/f1_msg_utils.h"
#include <Eigen/Geometry>
void f1_datalogger_ros::F1MsgUtils::doNothing()
{
    //I wasn't kidding, this does nothing.
}
f1_datalogger_msgs::msg::PacketHeader f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketHeader& header_data)
{
    f1_datalogger_msgs::msg::PacketHeader rtn;
    rtn.frame_identifier = header_data.m_frameIdentifier;
    rtn.packet_format = header_data.m_packetFormat;
    rtn.packet_id = header_data.m_packetId;
    rtn.packet_version = header_data.m_packetVersion;
    rtn.player_car_index = header_data.m_playerCarIndex;
    rtn.session_time = header_data.m_sessionTime;
    rtn.session_uid = header_data.m_sessionUID;

    return rtn;
}
f1_datalogger_msgs::msg::PacketMotionData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketMotionData& motion_data)
{
    f1_datalogger_msgs::msg::PacketMotionData rtn;
    rtn.header = toROS(motion_data.m_header);
    rtn.angular_acceleration.x = motion_data.m_angularAccelerationX;
    rtn.angular_acceleration.y = motion_data.m_angularAccelerationY;
    rtn.angular_acceleration.z = motion_data.m_angularAccelerationZ;
    rtn.angular_velocity.x = motion_data.m_angularVelocityX;
    rtn.angular_velocity.y = motion_data.m_angularVelocityY;
    rtn.angular_velocity.z = motion_data.m_angularVelocityZ;
    rtn.local_velocity.x = motion_data.m_localVelocityX;
    rtn.local_velocity.y = motion_data.m_localVelocityY;
    rtn.local_velocity.z = motion_data.m_localVelocityZ;
    rtn.front_wheels_angle = motion_data.m_frontWheelsAngle;
    std::copy(motion_data.m_wheelSlip,motion_data.m_wheelSlip+4, rtn.wheel_slip.begin());
    std::copy(motion_data.m_wheelSpeed,motion_data.m_wheelSpeed+4, rtn.wheel_speed.begin());
    std::copy(motion_data.m_suspensionAcceleration,motion_data.m_suspensionAcceleration+4, rtn.suspension_acceleration.begin());
    std::copy(motion_data.m_suspensionVelocity,motion_data.m_suspensionVelocity+4, rtn.suspension_velocity.begin());
    std::copy(motion_data.m_suspensionPosition,motion_data.m_suspensionPosition+4, rtn.suspension_position.begin());
    for (unsigned int i = 0; i < 20; i++)
    {
        rtn.car_motion_data[i] = toROS(motion_data.m_carMotionData[i]);
    }
    return rtn;
}
f1_datalogger_msgs::msg::CarMotionData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::CarMotionData& motion_data)
{
    
    f1_datalogger_msgs::msg::CarMotionData rtn;
    Eigen::Vector3d forwardVec( (double)motion_data.m_worldForwardDirX, (double)motion_data.m_worldForwardDirY, (double)motion_data.m_worldForwardDirZ );
    forwardVec.normalize();
    rtn.world_forward_dir.x = forwardVec.x();
    rtn.world_forward_dir.y = forwardVec.y();
    rtn.world_forward_dir.z = forwardVec.z();
    Eigen::Vector3d rightVec( (double)motion_data.m_worldRightDirX, (double)motion_data.m_worldRightDirY, (double)motion_data.m_worldRightDirZ );
    rightVec.normalize();
    Eigen::Vector3d leftVec( -rightVec );
    Eigen::Vector3d upVec = forwardVec.cross(leftVec);
    upVec.normalize();
    rtn.world_up_dir.x = upVec.x();
    rtn.world_up_dir.y = upVec.y();
    rtn.world_up_dir.z = upVec.z();
    rtn.world_right_dir.x = rightVec.x();
    rtn.world_right_dir.y = rightVec.y();
    rtn.world_right_dir.z = rightVec.z();

    rtn.world_position.x = motion_data.m_worldPositionX;
    rtn.world_position.y = motion_data.m_worldPositionY;
    rtn.world_position.z = motion_data.m_worldPositionZ;

    rtn.world_velocity.x = motion_data.m_worldVelocityX;
    rtn.world_velocity.y = motion_data.m_worldVelocityY;
    rtn.world_velocity.z = motion_data.m_worldVelocityZ;

    rtn.g_force_lateral = motion_data.m_gForceLateral;
    rtn.g_force_longitudinal = motion_data.m_gForceLongitudinal;
    rtn.g_force_vertical = motion_data.m_gForceVertical;

    rtn.roll = motion_data.m_roll;
    rtn.pitch = motion_data.m_pitch;
    rtn.yaw = motion_data.m_yaw;
    



    return rtn;
}