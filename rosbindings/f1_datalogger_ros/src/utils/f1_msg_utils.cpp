/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2011, Willow Garage, Inc.
*  Copyright (c) 2015, Tal Regev.
*  Copyright (c) 2018 Intel Corporation.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/
#include "f1_datalogger_ros/utils/f1_msg_utils.h"
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/core.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>
#include <exception>
#define BOOST_ENDIAN_DEPRECATED_NAMES
#include <boost/endian/endian.hpp>
namespace enc = sensor_msgs::image_encodings;
int f1_datalogger_ros::F1MsgUtils::getCvType(const std::string & encoding)
{
  // Check for the most common encodings first
  if (encoding == enc::BGR8) {return CV_8UC3;}
  if (encoding == enc::MONO8) {return CV_8UC1;}
  if (encoding == enc::RGB8) {return CV_8UC3;}
  if (encoding == enc::MONO16) {return CV_16UC1;}
  if (encoding == enc::BGR16) {return CV_16UC3;}
  if (encoding == enc::RGB16) {return CV_16UC3;}
  if (encoding == enc::BGRA8) {return CV_8UC4;}
  if (encoding == enc::RGBA8) {return CV_8UC4;}
  if (encoding == enc::BGRA16) {return CV_16UC4;}
  if (encoding == enc::RGBA16) {return CV_16UC4;}

  // For bayer, return one-channel
  if (encoding == enc::BAYER_RGGB8) {return CV_8UC1;}
  if (encoding == enc::BAYER_BGGR8) {return CV_8UC1;}
  if (encoding == enc::BAYER_GBRG8) {return CV_8UC1;}
  if (encoding == enc::BAYER_GRBG8) {return CV_8UC1;}
  if (encoding == enc::BAYER_RGGB16) {return CV_16UC1;}
  if (encoding == enc::BAYER_BGGR16) {return CV_16UC1;}
  if (encoding == enc::BAYER_GBRG16) {return CV_16UC1;}
  if (encoding == enc::BAYER_GRBG16) {return CV_16UC1;}

  // Miscellaneous
  if (encoding == enc::YUV422) {return CV_8UC2;}

  // Check all the generic content encodings
  std::cmatch m;

  if (std::regex_match(encoding.c_str(), m,
    std::regex("(8U|8S|16U|16S|32S|32F|64F)C([0-9]+)")))
  {
    return CV_MAKETYPE(depthStrToInt(m[1].str()), atoi(m[2].str().c_str()));
  }

  if (std::regex_match(encoding.c_str(), m,
    std::regex("(8U|8S|16U|16S|32S|32F|64F)")))
  {
    return CV_MAKETYPE(depthStrToInt(m[1].str()), 1);
  }

  throw std::runtime_error("Unrecognized image encoding [" + encoding + "]");
}
int f1_datalogger_ros::F1MsgUtils::depthStrToInt(const std::string& depth)
{
  if (depth == "8U") {
    return 0;
  } else if (depth == "8S") {
    return 1;
  } else if (depth == "16U") {
    return 2;
  } else if (depth == "16S") {
    return 3;
  } else if (depth == "32S") {
    return 4;
  } else if (depth == "32F") {
    return 5;
  }
  return 6;
}
enum Encoding { INVALID = -1, GRAY = 0, RGB, BGR, RGBA, BGRA, YUV422, BAYER_RGGB, BAYER_BGGR, BAYER_GBRG, BAYER_GRBG};
Encoding getEncoding(const std::string & encoding)
{
  if ((encoding == enc::MONO8) || (encoding == enc::MONO16)) {return Encoding::GRAY;}
  if ((encoding == enc::BGR8) || (encoding == enc::BGR16)) {return Encoding::BGR;}
  if ((encoding == enc::RGB8) || (encoding == enc::RGB16)) {return Encoding::RGB;}
  if ((encoding == enc::BGRA8) || (encoding == enc::BGRA16)) {return Encoding::BGRA;}
  if ((encoding == enc::RGBA8) || (encoding == enc::RGBA16)) {return Encoding::RGBA;}
  if (encoding == enc::YUV422) {return Encoding::YUV422;}

  if ((encoding == enc::BAYER_RGGB8) || (encoding == enc::BAYER_RGGB16)) {return Encoding::BAYER_RGGB;}
  if ((encoding == enc::BAYER_BGGR8) || (encoding == enc::BAYER_BGGR16)) {return Encoding::BAYER_BGGR;}
  if ((encoding == enc::BAYER_GBRG8) || (encoding == enc::BAYER_GBRG16)) {return Encoding::BAYER_GBRG;}
  if ((encoding == enc::BAYER_GRBG8) || (encoding == enc::BAYER_GRBG16)) {return Encoding::BAYER_GRBG;}

  // We don't support conversions to/from other types
  return Encoding::INVALID;
}
sensor_msgs::msg::Image f1_datalogger_ros::F1MsgUtils::toImageMsg(const cv::Mat & image, std_msgs::msg::Header header)
{
  sensor_msgs::msg::Image ros_image;
  ros_image.header = header;
  ros_image.height = image.rows;
  ros_image.width = image.cols;
  
  if(image.type()==CV_8UC3)
  {
    ros_image.encoding = "rgb8";
  }
  else
  {
    ros_image.encoding = "rgba8";
  }
  
  ros_image.is_bigendian = (boost::endian::order::native == boost::endian::order::big);
  ros_image.step = image.cols * image.elemSize();
  size_t size = ros_image.step * image.rows;
  ros_image.data.resize(size);

  if (image.isContinuous()) {
    memcpy(reinterpret_cast<char *>(&ros_image.data[0]), image.data, size);
  } else {
    // Copy by row by row
    uchar * ros_data_ptr = reinterpret_cast<uchar *>(&ros_image.data[0]);
    uchar * cv_data_ptr = image.data;
    for (int i = 0; i < image.rows; ++i) {
      memcpy(ros_data_ptr, cv_data_ptr, ros_image.step);
      ros_data_ptr += ros_image.step;
      cv_data_ptr += image.step;
    }
  }

  return ros_image;
}
void f1_datalogger_ros::F1MsgUtils::doNothing()
{
    //I wasn't kidding, this does nothing.
}
f1_datalogger_msgs::msg::MarshalZone f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::MarshalZone& marshal_zone)
{
  f1_datalogger_msgs::msg::MarshalZone rtn;
  rtn.zone_flag = marshal_zone.m_zoneFlag;
  rtn.zone_start = marshal_zone.m_zoneStart;
  return rtn;
}
f1_datalogger_msgs::msg::PacketSessionData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketSessionData& session_data)
{
  f1_datalogger_msgs::msg::PacketSessionData rtn;
  rtn.header=toROS(session_data.m_header);
  rtn.is_spectating = session_data.m_isSpectating;
  for (unsigned int i = 0; i < 21; i++)
  {
    rtn.marshal_zones[i] = toROS(session_data.m_marshalZones[i]);
  }
  rtn.network_game = session_data.m_networkGame;
  rtn.num_marshal_zones = session_data.m_numMarshalZones;
  rtn.pit_speed_limit = session_data.m_pitSpeedLimit;
  rtn.safety_car_status = session_data.m_safetyCarStatus;
  rtn.session_duration = session_data.m_sessionDuration;
  rtn.session_type = session_data.m_sessionType;
  rtn.session_time_left = session_data.m_sessionTimeLeft;
  rtn.game_paused  = session_data.m_gamePaused;
  rtn.era = session_data.m_era;
  rtn.air_temperature = session_data.m_airTemperature;
  return rtn;
}
f1_datalogger_msgs::msg::CarTelemetryData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::CarTelemetryData& telemetry_data)
{
  f1_datalogger_msgs::msg::CarTelemetryData  rtn;
  
  rtn.brake = telemetry_data.m_brake;
  rtn.clutch = telemetry_data.m_clutch;
  rtn.drs = telemetry_data.m_drs;
  rtn.engine_rpm = telemetry_data.m_engineRPM;
  rtn.engine_temperature = telemetry_data.m_engineTemperature;
  rtn.gear = telemetry_data.m_gear;
  rtn.rev_lights_percent = telemetry_data.m_revLightsPercent;
  rtn.speed = telemetry_data.m_speed;
  rtn.throttle = telemetry_data.m_throttle;
  std::copy(telemetry_data.m_brakesTemperature, telemetry_data.m_brakesTemperature+4, rtn.brakes_temperature.begin());
  std::copy(telemetry_data.m_tyresInnerTemperature, telemetry_data.m_tyresInnerTemperature+4, rtn.tyres_inner_temperature.begin());
  std::copy(telemetry_data.m_tyresPressure, telemetry_data.m_tyresPressure+4, rtn.tyres_pressure.begin());
  std::copy(telemetry_data.m_tyresSurfaceTemperature, telemetry_data.m_tyresSurfaceTemperature+4, rtn.tyres_surface_temperature.begin());
  return rtn;
}
f1_datalogger_msgs::msg::PacketCarTelemetryData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketCarTelemetryData& telemetry_data)
{
  f1_datalogger_msgs::msg::PacketCarTelemetryData rtn;
  rtn.button_status = telemetry_data.m_buttonStatus;
  rtn.header = toROS(telemetry_data.m_header);
  for(unsigned int i = 0; i < 20; i++)
  {
    rtn.car_telemetry_data[i] = toROS(telemetry_data.m_carTelemetryData[i]);
  }
  return rtn;
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
    rtn.world_forward_dir.header.frame_id=world_coordinate_name;
 //   rtn.world_forward_dir.header.stamp = rostime;
    rtn.world_forward_dir.vector.x = forwardVec.x();
    rtn.world_forward_dir.vector.y = forwardVec.y();
    rtn.world_forward_dir.vector.z = forwardVec.z();
    Eigen::Vector3d rightVec( (double)motion_data.m_worldRightDirX, (double)motion_data.m_worldRightDirY, (double)motion_data.m_worldRightDirZ );
    rightVec.normalize();
    Eigen::Vector3d leftVec( -rightVec );
    Eigen::Vector3d upVec = forwardVec.cross(leftVec);
    upVec.normalize();
    rtn.world_up_dir.header.frame_id=world_coordinate_name;
  //  rtn.world_up_dir.header.stamp = rostime;
    rtn.world_up_dir.vector.x = upVec.x();
    rtn.world_up_dir.vector.y = upVec.y();
    rtn.world_up_dir.vector.z = upVec.z();
    rtn.world_right_dir.header.frame_id=world_coordinate_name;
    //rtn.world_right_dir.header.stamp = rostime;
    rtn.world_right_dir.vector.x = rightVec.x();
    rtn.world_right_dir.vector.y = rightVec.y();
    rtn.world_right_dir.vector.z = rightVec.z();


    rtn.world_position.header.frame_id=world_coordinate_name;
 //   rtn.world_position.header.stamp = rostime;
    rtn.world_position.point.x = motion_data.m_worldPositionX;
    rtn.world_position.point.y = motion_data.m_worldPositionY;
    rtn.world_position.point.z = motion_data.m_worldPositionZ;


    rtn.world_velocity.header.frame_id=world_coordinate_name;
  //  rtn.world_velocity.header.stamp = rostime;
    rtn.world_velocity.vector.x = motion_data.m_worldVelocityX;
    rtn.world_velocity.vector.y = motion_data.m_worldVelocityY;
    rtn.world_velocity.vector.z = motion_data.m_worldVelocityZ;

    rtn.g_force_lateral = motion_data.m_gForceLateral;
    rtn.g_force_longitudinal = motion_data.m_gForceLongitudinal;
    rtn.g_force_vertical = motion_data.m_gForceVertical;

    rtn.roll = motion_data.m_roll;
    rtn.pitch = motion_data.m_pitch;
    rtn.yaw = motion_data.m_yaw;
    



    return rtn;
}