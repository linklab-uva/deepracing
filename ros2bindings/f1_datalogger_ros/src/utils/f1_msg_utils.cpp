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
#include <execution>
#include <exception>
#include <algorithm>
#define BOOST_ENDIAN_DEPRECATED_NAMES
#include <boost/endian/endian.hpp>

f1_datalogger_msgs::msg::MarshalZone f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::MarshalZone& marshal_zone)
{
  f1_datalogger_msgs::msg::MarshalZone rtn;
  rtn.zone_flag = marshal_zone.m_zoneFlag;
  rtn.zone_start = marshal_zone.m_zoneStart;
  return rtn;
}


f1_datalogger_msgs::msg::CarSetupData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::CarSetupData& setup_data)
{
  f1_datalogger_msgs::msg::CarSetupData rtn;
  rtn.ballast = setup_data.m_ballast;
  rtn.brake_bias = setup_data.m_brakeBias;
  rtn.brake_pressure = setup_data.m_brakePressure;
  rtn.front_anti_roll_bar = setup_data.m_frontAntiRollBar;
  rtn.front_camber = setup_data.m_frontCamber;
  rtn.front_suspension = setup_data.m_frontSuspension;
  rtn.front_suspension_height = setup_data.m_frontSuspensionHeight;
  rtn.front_toe = setup_data.m_frontToe;
  rtn.front_tyre_pressure = setup_data.m_frontTyrePressure;
  rtn.front_wing = setup_data.m_frontWing;
  rtn.fuel_load = setup_data.m_fuelLoad;
  rtn.off_throttle = setup_data.m_offThrottle;
  rtn.on_throttle = setup_data.m_onThrottle;
  rtn.rear_anti_roll_bar = setup_data.m_rearAntiRollBar;
  rtn.rear_camber = setup_data.m_rearCamber;
  rtn.rear_suspension = setup_data.m_rearSuspension;
  rtn.rear_suspension_height = setup_data.m_rearSuspensionHeight;
  rtn.rear_toe = setup_data.m_rearToe;
  rtn.rear_tyre_pressure = setup_data.m_rearTyrePressure;
  rtn.rear_wing = setup_data.m_rearWing;
  return rtn;
}
f1_datalogger_msgs::msg::PacketCarSetupData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketCarSetupData& packet_setup_data, bool copy_all_cars)
{
  f1_datalogger_msgs::msg::PacketCarSetupData rtn;
  rtn.header = f1_datalogger_ros::F1MsgUtils::toROS(packet_setup_data.m_header);
  if (rtn.header.player_car_index<20)
  {
    rtn.car_setup_data[rtn.header.player_car_index] = f1_datalogger_ros::F1MsgUtils::toROS(packet_setup_data.m_carSetups[rtn.header.player_car_index]);
  }
  if(copy_all_cars)
  {
    for(unsigned int i =0; i < 20; i++)
    {
      if (i!=rtn.header.player_car_index)
      {
        rtn.car_setup_data[i] = f1_datalogger_ros::F1MsgUtils::toROS(packet_setup_data.m_carSetups[i]);
      }
    }
  }
  return rtn;
}


f1_datalogger_msgs::msg::CarStatusData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::CarStatusData& status_data)
{
  f1_datalogger_msgs::msg::CarStatusData rtn;
  rtn.anti_lock_brakes = status_data.m_antiLockBrakes;
  rtn.drs_allowed = status_data.m_drsAllowed;
  rtn.engine_damage = status_data.m_engineDamage;
  rtn.engine_damage = status_data.m_engineDamage;
  rtn.ers_deploy_mode = status_data.m_ersDeployMode;
  rtn.ers_deployed_this_lap = status_data.m_ersDeployedThisLap;
  rtn.ers_harvested_this_lap_mguh = status_data.m_ersHarvestedThisLapMGUH;
  rtn.ers_harvested_this_lap_mguk = status_data.m_ersHarvestedThisLapMGUK;
  rtn.ers_store_energy = status_data.m_ersStoreEnergy;
  rtn.exhaust_damage = status_data.m_exhaustDamage;
  rtn.front_brake_bias = status_data.m_frontBrakeBias;
  rtn.front_left_wing_damage = status_data.m_frontLeftWingDamage;
  rtn.front_right_wing_damage = status_data.m_frontRightWingDamage;
  rtn.fuel_capacity = status_data.m_fuelCapacity;
  rtn.fuel_in_tank = status_data.m_fuelInTank;
  rtn.fuel_mix = status_data.m_fuelMix;
  rtn.gear_box_damage = status_data.m_gearBoxDamage;
  rtn.idle_rpm = status_data.m_idleRPM;
  rtn.max_gears = status_data.m_maxGears;
  rtn.max_rpm = status_data.m_maxRPM;
  rtn.pit_limiter_status = status_data.m_pitLimiterStatus;
  rtn.rear_wing_damage = status_data.m_rearWingDamage;
  rtn.traction_control = status_data.m_tractionControl;
  rtn.vehicle_fia_flags = status_data.m_vehicleFiaFlags;
  rtn.tyre_compound = status_data.m_tyreCompound;
  std::copy_n(&(status_data.m_tyresDamage[0]), 4, rtn.tyres_damage.begin());
  std::copy_n(&(status_data.m_tyresWear[0]), 4, rtn.tyres_wear.begin());
  return rtn;
}
f1_datalogger_msgs::msg::PacketCarStatusData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketCarStatusData& packet_status_data, bool copy_all_cars)
{

  f1_datalogger_msgs::msg::PacketCarStatusData rtn;
  rtn.header = f1_datalogger_ros::F1MsgUtils::toROS(packet_status_data.m_header);
  if (rtn.header.player_car_index<20)
  {
    rtn.car_status_data[rtn.header.player_car_index] = f1_datalogger_ros::F1MsgUtils::toROS(packet_status_data.m_carStatusData[rtn.header.player_car_index]);
  }
  if(copy_all_cars)
  {
    for(unsigned int i =0; i < 20; i++)
    {
      if (i!=rtn.header.player_car_index)
      {
        rtn.car_status_data[i] = f1_datalogger_ros::F1MsgUtils::toROS(packet_status_data.m_carStatusData[i]);
      }
    }
  }
  return rtn;
}


f1_datalogger_msgs::msg::PacketLapData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketLapData& lap_data, bool copy_all_cars)
{
  f1_datalogger_msgs::msg::PacketLapData rtn;
  rtn.header = f1_datalogger_ros::F1MsgUtils::toROS(lap_data.m_header);
  if (rtn.header.player_car_index<20)
  {
    rtn.lap_data[rtn.header.player_car_index] = f1_datalogger_ros::F1MsgUtils::toROS(lap_data.m_lapData[rtn.header.player_car_index]);
  }
  if(copy_all_cars)
  {
    for(unsigned int i =0; i < 20; i++)
    {
      if (i!=rtn.header.player_car_index)
      {
        rtn.lap_data[i] = f1_datalogger_ros::F1MsgUtils::toROS(lap_data.m_lapData[i]);
      }
    }
  }
  return rtn;
}
f1_datalogger_msgs::msg::LapData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::LapData& lap_data)
{
  f1_datalogger_msgs::msg::LapData rtn;
  rtn.best_lap_time = lap_data.m_bestLapTime;
  rtn.car_position = lap_data.m_carPosition;
  rtn.current_lap_invalid = lap_data.m_currentLapInvalid;
  rtn.current_lap_num = lap_data.m_currentLapNum;
  rtn.current_lap_time = lap_data.m_currentLapTime;
  rtn.driver_status = lap_data.m_driverStatus;
  rtn.grid_position = lap_data.m_gridPosition;
  rtn.lap_distance = lap_data.m_lapDistance;
  rtn.last_lap_time = lap_data.m_lastLapTime;
  rtn.penalties = lap_data.m_penalties;
  rtn.pit_status = lap_data.m_pitStatus;
  rtn.result_status = lap_data.m_resultStatus;
  rtn.safety_car_delta = lap_data.m_safetyCarDelta;
  rtn.sector1_time = lap_data.m_sector1Time;
  rtn.sector2_time = lap_data.m_sector2Time;
  rtn.sector = lap_data.m_sector;
  rtn.total_distance = lap_data.m_totalDistance;
  return rtn;
}

f1_datalogger_msgs::msg::PacketSessionData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketSessionData& session_data)
{
  f1_datalogger_msgs::msg::PacketSessionData rtn;
  rtn.header=toROS(session_data.m_header);
  for (unsigned int i = 0; i < 21; i++)
  {
    rtn.marshal_zones[i] = toROS(session_data.m_marshalZones[i]);
  }
  rtn.air_temperature = session_data.m_airTemperature;
  rtn.era = session_data.m_era;
  rtn.game_paused  = session_data.m_gamePaused;
  rtn.is_spectating = session_data.m_isSpectating;
  rtn.network_game = session_data.m_networkGame;
  rtn.num_marshal_zones = session_data.m_numMarshalZones;
  rtn.pit_speed_limit = session_data.m_pitSpeedLimit;
  rtn.safety_car_status = session_data.m_safetyCarStatus;
  rtn.session_duration = session_data.m_sessionDuration;
  rtn.session_type = session_data.m_sessionType;
  rtn.session_time_left = session_data.m_sessionTimeLeft;
  rtn.spectator_car_index = session_data.m_spectatorCarIndex;
  rtn.track_id = session_data.m_trackId;
  rtn.track_length = session_data.m_trackLength;
  rtn.track_temperature = session_data.m_trackTemperature;
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
  rtn.steer = telemetry_data.m_steer;
  rtn.throttle = telemetry_data.m_throttle;
  std::copy_n(&(telemetry_data.m_brakesTemperature[0]), 4, rtn.brakes_temperature.begin());
  std::copy_n(&(telemetry_data.m_tyresInnerTemperature[0]), 4, rtn.tyres_inner_temperature.begin());
  std::copy_n(&(telemetry_data.m_tyresPressure[0]), 4, rtn.tyres_pressure.begin());
  std::copy_n(&(telemetry_data.m_tyresSurfaceTemperature[0]), 4, rtn.tyres_surface_temperature.begin());
  return rtn;
}
f1_datalogger_msgs::msg::PacketCarTelemetryData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketCarTelemetryData& telemetry_data, bool copy_all_cars)
{
  f1_datalogger_msgs::msg::PacketCarTelemetryData rtn;
  rtn.header = f1_datalogger_ros::F1MsgUtils::toROS(telemetry_data.m_header);
  if (rtn.header.player_car_index<20)
  {
    rtn.car_telemetry_data[rtn.header.player_car_index] = f1_datalogger_ros::F1MsgUtils::toROS(telemetry_data.m_carTelemetryData[rtn.header.player_car_index]);
  }
  if(copy_all_cars)
  {
    for(unsigned int i =0; i < 20; i++)
    {
      if (i!=rtn.header.player_car_index)
      {
        rtn.car_telemetry_data[i] = f1_datalogger_ros::F1MsgUtils::toROS(telemetry_data.m_carTelemetryData[i]);
      }
    }
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
f1_datalogger_msgs::msg::PacketMotionData f1_datalogger_ros::F1MsgUtils::toROS(const deepf1::twenty_eighteen::PacketMotionData& motion_data, bool copy_all_cars)
{
    f1_datalogger_msgs::msg::PacketMotionData rtn;
    rtn.header = f1_datalogger_ros::F1MsgUtils::toROS(motion_data.m_header);
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
    std::copy_n(std::cbegin(motion_data.m_wheelSlip), 4, rtn.wheel_slip.begin());
    std::copy_n(std::cbegin(motion_data.m_wheelSpeed), 4, rtn.wheel_speed.begin());
    std::copy_n(std::cbegin(motion_data.m_suspensionAcceleration), 4, rtn.suspension_acceleration.begin());
    std::copy_n(std::cbegin(motion_data.m_suspensionVelocity), 4, rtn.suspension_velocity.begin());
    std::copy_n(std::cbegin(motion_data.m_suspensionPosition), 4, rtn.suspension_position.begin());
    
    if(copy_all_cars)
    {
      auto beg = std::cbegin<deepf1::twenty_eighteen::CarMotionData [20]>(motion_data.m_carMotionData);
      auto end = std::cend<deepf1::twenty_eighteen::CarMotionData [20]>(motion_data.m_carMotionData);
      std::function<f1_datalogger_msgs::msg::CarMotionData (const deepf1::twenty_eighteen::CarMotionData&)> f = &f1_datalogger_ros::F1MsgUtils::toROSMotionData;
      std::transform(beg, end, rtn.car_motion_data.begin(), f);
    }
    else if(rtn.header.player_car_index<20)
    {
      rtn.car_motion_data[rtn.header.player_car_index] = f1_datalogger_ros::F1MsgUtils::toROSMotionData(motion_data.m_carMotionData[rtn.header.player_car_index]);
    }
    return rtn;
}
f1_datalogger_msgs::msg::CarMotionData f1_datalogger_ros::F1MsgUtils::toROSMotionData(const deepf1::twenty_eighteen::CarMotionData& motion_data)
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