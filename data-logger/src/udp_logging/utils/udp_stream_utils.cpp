#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"
#include <thread>
namespace deepf1
{ 


UDPStreamUtils::UDPStreamUtils()
{
}


UDPStreamUtils::~UDPStreamUtils()
{
}

deepf1::protobuf::CarUDPData UDPStreamUtils::toProto(const deepf1::CarUDPData2017& fromStream)
{
	
    deepf1::protobuf::CarUDPData rtn;
	
	rtn.mutable_m_worldposition()->Resize(3, -1.0);
	memcpy(rtn.mutable_m_worldposition()->mutable_data(),fromStream.m_worldPosition,3*sizeof(float));

    rtn.set_m_lastlaptime(fromStream.m_lastLapTime);

    rtn.set_m_currentlaptime(fromStream.m_currentLapTime);

    rtn.set_m_bestlaptime(fromStream.m_bestLapTime);
    
    rtn.set_m_sector1time(fromStream.m_sector1Time);

    rtn.set_m_sector2time(fromStream.m_sector2Time);

    rtn.set_m_lapdistance(fromStream.m_lapDistance);

    rtn.set_m_driverid(fromStream.m_driverId);

    rtn.set_m_teamid(fromStream.m_teamId);

    rtn.set_m_carposition(fromStream.m_carPosition);
    
    rtn.set_m_currentlapnum(fromStream.m_currentLapNum);

    rtn.set_m_tyrecompound(fromStream.m_tyreCompound);

    rtn.set_m_inpits(fromStream.m_inPits);

    rtn.set_m_sector(fromStream.m_sector);

    rtn.set_m_currentlapinvalid(fromStream.m_currentLapInvalid);

    rtn.set_m_penalties(fromStream.m_penalties);

    return rtn;
}
deepf1::protobuf::UDPData UDPStreamUtils::toProto(const deepf1::UDPPacket2017& fromStream)
{
    deepf1::protobuf::UDPData rtn;

    rtn.set_m_ang_acc_z(fromStream.m_ang_acc_z);

    rtn.set_m_ang_acc_y(fromStream.m_ang_acc_y);

    rtn.set_m_ang_acc_x(fromStream.m_ang_acc_x);

    rtn.set_m_z_local_velocity(fromStream.m_z_local_velocity);

    rtn.set_m_y_local_velocity(fromStream.m_y_local_velocity);

    rtn.set_m_x_local_velocity(fromStream.m_x_local_velocity);

    rtn.set_m_roll(fromStream.m_roll);

    rtn.set_m_pitch(fromStream.m_pitch);

    rtn.set_m_yaw(fromStream.m_yaw);

	rtn.set_m_player_car_index(fromStream.m_player_car_index);

	rtn.set_m_num_cars(fromStream.m_num_cars);

	rtn.set_m_spectator_car_index(fromStream.m_spectator_car_index);

	rtn.set_m_is_spectating(fromStream.m_is_spectating);

	rtn.set_m_rev_lights_percent(fromStream.m_rev_lights_percent);

	rtn.set_m_session_time_left(fromStream.m_session_time_left);

	rtn.set_m_pit_speed_limit(fromStream.m_pit_speed_limit);

	rtn.set_m_pit_limiter_status(fromStream.m_pit_limiter_status);

	rtn.set_m_exhaust_damage(fromStream.m_exhaust_damage);

	rtn.set_m_gear_box_damage(fromStream.m_gear_box_damage);

	rtn.set_m_engine_damage(fromStream.m_engine_damage);

	rtn.set_m_rear_wing_damage(fromStream.m_rear_wing_damage);

	rtn.set_m_front_right_wing_damage(fromStream.m_front_right_wing_damage);

	rtn.set_m_front_left_wing_damage(fromStream.m_front_left_wing_damage);

	rtn.set_m_currentlapinvalid(fromStream.m_currentLapInvalid);

	rtn.set_m_fuel_mix(fromStream.m_fuel_mix);

	rtn.set_m_front_brake_bias(fromStream.m_front_brake_bias);

	rtn.set_m_tyre_compound(fromStream.m_tyre_compound);

	rtn.set_m_ang_vel_z(fromStream.m_ang_vel_z);

	rtn.set_m_ang_vel_y(fromStream.m_ang_vel_y);

	rtn.set_m_ang_vel_x(fromStream.m_ang_vel_x);
	
	rtn.set_m_gforce_vert(fromStream.m_gforce_vert);

	rtn.set_m_engine_temperature(fromStream.m_engine_temperature);

	rtn.set_m_era(fromStream.m_era);

	rtn.set_m_vehiclefiaflags(fromStream.m_vehicleFIAFlags);

	rtn.set_m_track_number(fromStream.m_track_number);

	rtn.set_m_drsallowed(fromStream.m_drsAllowed);

	rtn.set_m_sessiontype(fromStream.m_sessionType);

	rtn.set_m_max_gears(fromStream.m_max_gears);

	rtn.set_m_idle_rpm(fromStream.m_idle_rpm);

	rtn.set_m_max_rpm(fromStream.m_max_rpm);

	rtn.set_m_last_lap_time(fromStream.m_last_lap_time);

	rtn.set_m_track_size(fromStream.m_track_size);

	rtn.set_m_total_laps(fromStream.m_total_laps);

	rtn.set_m_team_info(fromStream.m_team_info);

	rtn.set_m_sector2_time(fromStream.m_sector2_time);

	rtn.set_m_sector1_time(fromStream.m_sector1_time);

	rtn.set_m_sector(fromStream.m_sector);

	rtn.set_m_in_pits(fromStream.m_in_pits);

	rtn.set_m_fuel_capacity(fromStream.m_fuel_capacity);

	rtn.set_m_fuel_in_tank(fromStream.m_fuel_in_tank);

	rtn.set_m_anti_lock_brakes(fromStream.m_anti_lock_brakes);

	rtn.set_m_traction_control(fromStream.m_traction_control);

	rtn.set_m_drs(fromStream.m_drs);

	rtn.set_m_kers_max_level(fromStream.m_kers_max_level);

	rtn.set_m_kers_level(fromStream.m_kers_level);

	rtn.set_m_car_position(fromStream.m_car_position);

	rtn.set_m_sli_pro_native_support(fromStream.m_sli_pro_native_support);

	rtn.set_m_enginerate(fromStream.m_engineRate);

	rtn.set_m_lap(fromStream.m_lap);

	rtn.set_m_gforce_lon(fromStream.m_gforce_lon);

	rtn.set_m_gforce_lat(fromStream.m_gforce_lat);

	rtn.set_m_gear(fromStream.m_gear);

	rtn.set_m_brake(fromStream.m_brake);

	rtn.set_m_steer(fromStream.m_steer);

	rtn.set_m_throttle(fromStream.m_throttle);

	rtn.set_m_zd(fromStream.m_zd);

	rtn.set_m_yd(fromStream.m_yd);

	rtn.set_m_xd(fromStream.m_xd);

	rtn.set_m_zr(fromStream.m_zr);

	rtn.set_m_yr(fromStream.m_yr);

	rtn.set_m_xr(fromStream.m_xr);

	rtn.set_m_zv(fromStream.m_zv);

	rtn.set_m_yv(fromStream.m_yv);

	rtn.set_m_xv(fromStream.m_xv);

	rtn.set_m_speed(fromStream.m_speed);

	rtn.set_m_z(fromStream.m_z);

	rtn.set_m_y(fromStream.m_y);

	rtn.set_m_x(fromStream.m_x);

	rtn.set_m_totaldistance(fromStream.m_totalDistance);

	rtn.set_m_lapdistance(fromStream.m_lapDistance);

	rtn.set_m_laptime(fromStream.m_lapTime);

	rtn.set_m_time(fromStream.m_time);
	
	rtn.mutable_m_wheel_speed()->Resize(4,-1.0);
	memcpy(rtn.mutable_m_wheel_speed()->mutable_data(),fromStream.m_wheel_speed,4*sizeof(float));
	
	rtn.mutable_m_susp_vel()->Resize(4,-1.0);
	memcpy(rtn.mutable_m_susp_vel()->mutable_data(),fromStream.m_susp_vel,4*sizeof(float));
	
	rtn.mutable_m_susp_pos()->Resize(4,-1.0);
	memcpy(rtn.mutable_m_susp_pos()->mutable_data(),fromStream.m_susp_pos,4*sizeof(float));
	
	rtn.mutable_m_susp_acceleration()->Resize(4,-1.0);
	memcpy(rtn.mutable_m_susp_acceleration()->mutable_data(),fromStream.m_susp_acceleration,4*sizeof(float));
	
	rtn.mutable_m_tyres_pressure()->Resize(4, -1.0);
	memcpy(rtn.mutable_m_tyres_pressure()->mutable_data(),fromStream.m_tyres_pressure,4*sizeof(float));
	
	rtn.mutable_m_brakes_temp()->Resize(4, -1.0);
	memcpy(rtn.mutable_m_brakes_temp()->mutable_data(),fromStream.m_brakes_temp,4*sizeof(float));


	rtn.mutable_m_tyres_damage()->Resize(4, -127);
	rtn.mutable_m_tyres_wear()->Resize(4, -127);
	rtn.mutable_m_tyres_temperature()->Resize(4, -127);
    for(unsigned int i = 0 ; i < 4 ; i++)
    {
		rtn.mutable_m_tyres_damage()->Set(i,(google::protobuf::uint32)fromStream.m_tyres_damage[i]);
		rtn.mutable_m_tyres_wear()->Set(i,(google::protobuf::uint32)fromStream.m_tyres_wear[i]);
		rtn.mutable_m_tyres_temperature()->Set(i,(google::protobuf::uint32)fromStream.m_tyres_temperature[i]);
    }
	rtn.mutable_m_car_data()->Reserve(20);
    for(unsigned int i = 0 ; i < 20 ; i++)
    {
		deepf1::protobuf::CarUDPData* added = rtn.add_m_car_data();
		const deepf1::protobuf::CarUDPData fromstream = toProto(fromStream.m_car_data[i]);
		added->CopyFrom(fromstream);
    }

    return rtn;
}

namespace twenty_eighteen
{
	TwentyEighteenUDPStreamUtils::TwentyEighteenUDPStreamUtils()
	{

	}
	TwentyEighteenUDPStreamUtils::~TwentyEighteenUDPStreamUtils()
	{
		
	}
	
	deepf1::twenty_eighteen::protobuf::PacketCarStatusData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketCarStatusData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketCarStatusData rtn;
		rtn.mutable_m_header()->CopyFrom(toProto(fromStream.m_header));
		unsigned int num_cars = 20;
		rtn.mutable_m_carstatusdata()->Reserve(num_cars);
		for(unsigned int i = 0; i < num_cars; i++)
		{
			deepf1::twenty_eighteen::protobuf::CarStatusData* added = rtn.add_m_carstatusdata();
			deepf1::twenty_eighteen::protobuf::CarStatusData fromstream = toProto(fromStream.m_carStatusData[i]);
			added->CopyFrom(fromstream);
		}
		
		return rtn;
	}
	deepf1::twenty_eighteen::protobuf::PacketCarSetupData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketCarSetupData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketCarSetupData rtn;
		rtn.mutable_m_header()->CopyFrom(toProto(fromStream.m_header));
		unsigned int num_cars = 20;
		rtn.mutable_m_carsetups()->Reserve(num_cars);
		for(unsigned int i = 0; i < num_cars; i++)
		{
			deepf1::twenty_eighteen::protobuf::CarSetupData* added = rtn.add_m_carsetups();
			deepf1::twenty_eighteen::protobuf::CarSetupData fromstream = toProto(fromStream.m_carSetups[i]);
			added->CopyFrom(fromstream);
		}
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::PacketCarTelemetryData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketCarTelemetryData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketCarTelemetryData rtn;
		rtn.mutable_m_header()->CopyFrom(toProto(fromStream.m_header));
		unsigned int num_cars = 20;
		rtn.mutable_m_cartelemetrydata()->Reserve(num_cars);
		for(unsigned int i = 0; i < num_cars; i++)
		{
			deepf1::twenty_eighteen::protobuf::CarTelemetryData* added = rtn.add_m_cartelemetrydata();
			deepf1::twenty_eighteen::protobuf::CarTelemetryData fromstream = toProto(fromStream.m_carTelemetryData[i]);
			added->CopyFrom(fromstream);
		}
		rtn.set_m_buttonstatus(fromStream.m_buttonStatus);
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::PacketEventData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketEventData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketEventData rtn;
		rtn.mutable_m_header()->CopyFrom(toProto(fromStream.m_header));
		char code[4];
		for(unsigned int i = 0; i < 4; i++)
		{
			code[i] = fromStream.m_eventStringCode[i];
		}
		rtn.set_m_eventstringcode(std::string(code));
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::PacketLapData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketLapData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketLapData rtn;
		rtn.mutable_m_header()->CopyFrom(toProto(fromStream.m_header));
		unsigned int num_cars = 20;
		rtn.mutable_m_lapdata()->Reserve(num_cars);
		for(unsigned int i = 0; i < num_cars; i++)
		{
			deepf1::twenty_eighteen::protobuf::LapData* added = rtn.add_m_lapdata();
			deepf1::twenty_eighteen::protobuf::LapData fromstream = toProto(fromStream.m_lapData[i]);
			added->CopyFrom(fromstream);
		}
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::PacketMotionData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketMotionData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketMotionData rtn;
		rtn.mutable_m_header()->CopyFrom(toProto(fromStream.m_header));
		unsigned int num_cars = 20;
		rtn.mutable_m_carmotiondata()->Reserve(num_cars);
		for(unsigned int i = 0; i < num_cars; i++)
		{
			deepf1::twenty_eighteen::protobuf::CarMotionData* added = rtn.add_m_carmotiondata();
			deepf1::twenty_eighteen::protobuf::CarMotionData fromstream = toProto(fromStream.m_carMotionData[i]);
			added->CopyFrom(fromstream);
		}

		rtn.mutable_m_suspensionacceleration()->Resize(4,-1E5);
		rtn.mutable_m_suspensionposition()->Resize(4,-1E5);
		rtn.mutable_m_suspensionvelocity()->Resize(4,-1E5);
		rtn.mutable_m_wheelslip()->Resize(4,-1E5);
		rtn.mutable_m_wheelspeed()->Resize(4,-1E5);
		
		
		memcpy(rtn.mutable_m_suspensionacceleration()->mutable_data(), fromStream.m_suspensionAcceleration, 4*sizeof(float));
		memcpy(rtn.mutable_m_suspensionposition()->mutable_data(), fromStream.m_suspensionPosition, 4*sizeof(float));
		memcpy(rtn.mutable_m_suspensionvelocity()->mutable_data(), fromStream.m_suspensionVelocity, 4*sizeof(float));
		memcpy(rtn.mutable_m_wheelslip()->mutable_data(), fromStream.m_wheelSlip, 4*sizeof(float));
		memcpy(rtn.mutable_m_wheelspeed()->mutable_data(), fromStream.m_wheelSpeed, 4*sizeof(float));

		rtn.set_m_angularaccelerationx(fromStream.m_angularAccelerationX);
		rtn.set_m_angularaccelerationy(fromStream.m_angularAccelerationY);
		rtn.set_m_angularaccelerationz(fromStream.m_angularAccelerationZ);
		rtn.set_m_angularvelocityx(fromStream.m_angularVelocityX);
		rtn.set_m_angularvelocityy(fromStream.m_angularVelocityY);
		rtn.set_m_angularvelocityz(fromStream.m_angularVelocityZ);
		rtn.set_m_frontwheelsangle(fromStream.m_frontWheelsAngle);
		rtn.set_m_localvelocityx(fromStream.m_localVelocityX);
		rtn.set_m_localvelocityy(fromStream.m_localVelocityY);
		rtn.set_m_localvelocityz(fromStream.m_localVelocityZ);

		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::PacketParticipantsData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketParticipantsData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketParticipantsData rtn;
		rtn.mutable_m_header()->CopyFrom(toProto(fromStream.m_header));
		unsigned int num_cars = 20;
		rtn.set_m_numcars(fromStream.m_numCars);
		rtn.mutable_m_participants()->Reserve(num_cars);
		for(unsigned int i = 0; i < num_cars; i++)
		{
			deepf1::twenty_eighteen::protobuf::ParticipantData* added = rtn.add_m_participants();
			deepf1::twenty_eighteen::protobuf::ParticipantData fromstream = toProto(fromStream.m_participants[i]);
			added->CopyFrom(fromstream);
		}
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::PacketSessionData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketSessionData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketSessionData rtn;
		rtn.mutable_m_header()->CopyFrom(toProto(fromStream.m_header));
		unsigned int num_zones = 21;
		rtn.mutable_m_marshalzones()->Reserve(num_zones);
		for(unsigned int i = 0; i < num_zones; i++)
		{
			deepf1::twenty_eighteen::protobuf::MarshalZone* added = rtn.add_m_marshalzones();
			deepf1::twenty_eighteen::protobuf::MarshalZone fromstream = toProto(fromStream.m_marshalZones[i]);
			added->CopyFrom(fromstream);
		}
		rtn.set_m_airtemperature(fromStream.m_airTemperature);
		rtn.set_m_era(fromStream.m_era);
		rtn.set_m_gamepaused(fromStream.m_gamePaused);
		rtn.set_m_isspectating(fromStream.m_isSpectating);
		rtn.set_m_networkgame(fromStream.m_networkGame);
		rtn.set_m_nummarshalzones(fromStream.m_numMarshalZones);
		rtn.set_m_pitspeedlimit(fromStream.m_pitSpeedLimit);
		rtn.set_m_safetycarstatus(fromStream.m_safetyCarStatus);
		rtn.set_m_sessionduration(fromStream.m_sessionDuration);
		rtn.set_m_sessiontimeleft(fromStream.m_sessionTimeLeft);
		rtn.set_m_sessiontype(fromStream.m_sessionType);
		rtn.set_m_slipronativesupport(fromStream.m_spectatorCarIndex);
		rtn.set_m_totallaps(fromStream.m_totalLaps);
		rtn.set_m_trackid(fromStream.m_trackId);
		rtn.set_m_tracklength(fromStream.m_trackLength);
		rtn.set_m_tracktemperature(fromStream.m_trackTemperature);
		rtn.set_m_weather(fromStream.m_weather);
		return rtn;		
	}

	deepf1::twenty_eighteen::protobuf::PacketHeader TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::PacketHeader& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::PacketHeader rtn;
		rtn.set_m_frameidentifier(fromStream.m_frameIdentifier);
		rtn.set_m_packetformat(fromStream.m_packetFormat);
		rtn.set_m_packetid(google::protobuf::uint32(fromStream.m_packetId));
		rtn.set_m_packetversion(fromStream.m_packetVersion);
		rtn.set_m_playercarindex(fromStream.m_playerCarIndex);
		rtn.set_m_sessiontime(fromStream.m_sessionTime);
		rtn.set_m_sessionuid(fromStream.m_sessionUID);
		return rtn;
	}


	deepf1::twenty_eighteen::protobuf::CarStatusData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::CarStatusData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::CarStatusData rtn;
		rtn.mutable_m_tyresdamage()->Resize(4, 255);
		rtn.mutable_m_tyreswear()->Resize(4, 255);
		for(unsigned int i = 0 ; i < 4 ; i++)
		{
			rtn.mutable_m_tyresdamage()->Set(i,(google::protobuf::uint32)fromStream.m_tyresDamage[i]);
			rtn.mutable_m_tyreswear()->Set(i,(google::protobuf::uint32)fromStream.m_tyresWear[i]);
		}
		rtn.set_m_antilockbrakes(fromStream.m_antiLockBrakes);
		rtn.set_m_drsallowed(fromStream.m_drsAllowed);
		rtn.set_m_enginedamage(fromStream.m_engineDamage);
		rtn.set_m_ersdeployedthislap(fromStream.m_ersDeployedThisLap);
		rtn.set_m_ersdeploymode(fromStream.m_ersDeployMode);
		rtn.set_m_ersharvestedthislapmguh(fromStream.m_ersHarvestedThisLapMGUH);
		rtn.set_m_ersharvestedthislapmguk(fromStream.m_ersHarvestedThisLapMGUK);
		rtn.set_m_ersstoreenergy(fromStream.m_ersStoreEnergy);
		rtn.set_m_exhaustdamage(fromStream.m_exhaustDamage);
		rtn.set_m_frontbrakebias(fromStream.m_frontBrakeBias);
		rtn.set_m_frontleftwingdamage(fromStream.m_frontLeftWingDamage);
		rtn.set_m_frontrightwingdamage(fromStream.m_frontRightWingDamage);
		rtn.set_m_fuelcapacity(fromStream.m_fuelCapacity);
		rtn.set_m_fuelintank(fromStream.m_fuelInTank);
		rtn.set_m_fuelmix(fromStream.m_fuelMix);
		rtn.set_m_gearboxdamage(fromStream.m_gearBoxDamage);
		rtn.set_m_idlerpm(fromStream.m_idleRPM);
		rtn.set_m_maxgears(fromStream.m_maxGears);
		rtn.set_m_maxrpm(fromStream.m_maxRPM);
		rtn.set_m_pitlimiterstatus(fromStream.m_pitLimiterStatus);
		rtn.set_m_rearwingdamage(fromStream.m_rearWingDamage);
		rtn.set_m_tractioncontrol(fromStream.m_tractionControl);
		rtn.set_m_tyrecompound(fromStream.m_tyreCompound);
		rtn.set_m_vehiclefiaflags(fromStream.m_vehicleFiaFlags);
		return rtn;
	}
	deepf1::twenty_eighteen::protobuf::CarSetupData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::CarSetupData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::CarSetupData rtn;
		rtn.set_m_ballast(fromStream.m_ballast);
		rtn.set_m_brakebias(fromStream.m_brakeBias);
		rtn.set_m_brakepressure(fromStream.m_brakePressure);
		rtn.set_m_frontantirollbar(fromStream.m_frontAntiRollBar);
		rtn.set_m_frontcamber(fromStream.m_frontCamber);
		rtn.set_m_frontsuspension(fromStream.m_frontSuspension);
		rtn.set_m_frontsuspensionheight(fromStream.m_frontSuspensionHeight);
		rtn.set_m_fronttoe(fromStream.m_frontToe);
		rtn.set_m_fronttyrepressure(fromStream.m_frontTyrePressure);
		rtn.set_m_frontwing(fromStream.m_frontWing);
		rtn.set_m_fuelload(fromStream.m_fuelLoad);
		rtn.set_m_offthrottle(fromStream.m_offThrottle);
		rtn.set_m_onthrottle(fromStream.m_onThrottle);
		rtn.set_m_rearantirollbar(fromStream.m_rearAntiRollBar);
		rtn.set_m_rearcamber(fromStream.m_rearCamber);
		rtn.set_m_rearsuspension(fromStream.m_rearSuspension);
		rtn.set_m_rearsuspensionheight(fromStream.m_rearSuspensionHeight);
		rtn.set_m_reartoe(fromStream.m_rearToe);
		rtn.set_m_reartyrepressure(fromStream.m_rearTyrePressure);
		rtn.set_m_rearwing(fromStream.m_rearWing);
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::CarTelemetryData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::CarTelemetryData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::CarTelemetryData rtn;
		rtn.mutable_m_brakestemperature()->Resize(4,4294967295);
		rtn.mutable_m_tyresinnertemperature()->Resize(4,4294967295);
		rtn.mutable_m_tyrespressure()->Resize(4,4294967295);
		rtn.mutable_m_tyressurfacetemperature()->Resize(4,4294967295);
		for(unsigned int i = 0 ; i < 4 ; i++)
		{
			rtn.mutable_m_brakestemperature()->Set(i,(google::protobuf::uint32)fromStream.m_brakesTemperature[i]);
			rtn.mutable_m_tyresinnertemperature()->Set(i,(google::protobuf::uint32)fromStream.m_tyresInnerTemperature[i]);
			rtn.mutable_m_tyrespressure()->Set(i,(google::protobuf::uint32)fromStream.m_tyresPressure[i]);
			rtn.mutable_m_tyressurfacetemperature()->Set(i,(google::protobuf::uint32)fromStream.m_tyresSurfaceTemperature[i]);
		}
		rtn.set_m_brake(fromStream.m_brake);
		rtn.set_m_clutch(fromStream.m_clutch);
		rtn.set_m_drs(fromStream.m_drs);
		rtn.set_m_enginerpm(fromStream.m_engineRPM);
		rtn.set_m_enginetemperature(fromStream.m_engineTemperature);
		rtn.set_m_gear(fromStream.m_gear);
		rtn.set_m_revlightspercent(fromStream.m_revLightsPercent);
		rtn.set_m_speed(fromStream.m_speed);
		rtn.set_m_steer(fromStream.m_steer);
		rtn.set_m_throttle(fromStream.m_throttle);
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::LapData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::LapData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::LapData rtn;
		rtn.set_m_bestlaptime(fromStream.m_bestLapTime);
		rtn.set_m_carposition(fromStream.m_carPosition);
		rtn.set_m_currentlapinvalid(fromStream.m_currentLapInvalid);
		rtn.set_m_currentlapnum(fromStream.m_currentLapNum);
		rtn.set_m_currentlaptime(fromStream.m_currentLapTime);
		rtn.set_m_driverstatus(fromStream.m_driverStatus);
		rtn.set_m_gridposition(fromStream.m_gridPosition);
		rtn.set_m_lapdistance(fromStream.m_lapDistance);
		rtn.set_m_lastlaptime(fromStream.m_lastLapTime);
		rtn.set_m_penalties(fromStream.m_penalties);
		rtn.set_m_pitstatus(fromStream.m_pitStatus);
		rtn.set_m_resultstatus(fromStream.m_resultStatus);
		rtn.set_m_safetycardelta(fromStream.m_safetyCarDelta);
		rtn.set_m_sector1time(fromStream.m_sector1Time);
		rtn.set_m_sector2time(fromStream.m_sector2Time);
		rtn.set_m_sector(fromStream.m_sector);
		rtn.set_m_totaldistance(fromStream.m_totalDistance);
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::CarMotionData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::CarMotionData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::CarMotionData rtn;
		rtn.set_m_gforcelateral(fromStream.m_gForceLateral);
		rtn.set_m_gforcelongitudinal(fromStream.m_gForceLongitudinal);
		rtn.set_m_gforcevertical(fromStream.m_gForceVertical);
		rtn.set_m_pitch(fromStream.m_pitch);
		rtn.set_m_roll(fromStream.m_roll);
		rtn.set_m_worldforwarddirx(fromStream.m_worldForwardDirX);
		rtn.set_m_worldforwarddiry(fromStream.m_worldForwardDirY);
		rtn.set_m_worldforwarddirz(fromStream.m_worldForwardDirZ);
		rtn.set_m_worldpositionx(fromStream.m_worldPositionX);
		rtn.set_m_worldpositiony(fromStream.m_worldPositionY);
		rtn.set_m_worldpositionz(fromStream.m_worldPositionZ);
		rtn.set_m_worldrightdirx(fromStream.m_worldRightDirX);
		rtn.set_m_worldrightdiry(fromStream.m_worldRightDirY);
		rtn.set_m_worldrightdirz(fromStream.m_worldRightDirZ);
		rtn.set_m_worldvelocityx(fromStream.m_worldVelocityX);
		rtn.set_m_worldvelocityy(fromStream.m_worldVelocityY);
		rtn.set_m_worldvelocityz(fromStream.m_worldVelocityZ);
		rtn.set_m_yaw(fromStream.m_yaw);
		return rtn;		
	}
	deepf1::twenty_eighteen::protobuf::ParticipantData TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::ParticipantData& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::ParticipantData rtn;
		rtn.set_m_aicontrolled(fromStream.m_aiControlled);
		rtn.set_m_driverid(fromStream.m_driverId);
		rtn.set_m_name(std::string(fromStream.m_name));
		rtn.set_m_nationality(fromStream.m_nationality);
		rtn.set_m_racenumber(fromStream.m_raceNumber);
		rtn.set_m_teamid(fromStream.m_teamId);
		return rtn;		
	}
	
	deepf1::twenty_eighteen::protobuf::MarshalZone TwentyEighteenUDPStreamUtils::toProto(const deepf1::twenty_eighteen::MarshalZone& fromStream)
	{
		deepf1::twenty_eighteen::protobuf::MarshalZone rtn;
		rtn.set_m_zoneflag(fromStream.m_zoneFlag);
		rtn.set_m_zonestart(fromStream.m_zoneStart);
		return rtn;
	}

}

}