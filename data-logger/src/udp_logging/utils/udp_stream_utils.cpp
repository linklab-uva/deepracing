#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"
namespace deepf1
{ 


UDPStreamUtils::UDPStreamUtils()
{
}


UDPStreamUtils::~UDPStreamUtils()
{
}
/*
message CarUDPData {
        repeated float m_worldPosition = 1; // world co-ordinates of vehicle

		float m_lastLapTime = 2;

		float m_currentLapTime = 3;

		float m_bestLapTime = 4;

		float m_sector1Time = 5;

		float m_sector2Time = 6;

		float m_lapDistance = 7;

		uint32 m_driverId = 8;

		uint32 m_teamId = 9;

		uint32 m_carPosition = 10;     // UPDATED: track positions of vehicle

		uint32 m_currentLapNum = 11;

		uint32 m_tyreCompound = 12; // compound of tyre � 0 = ultra soft, 1 = super soft, 2 = soft, 3 = medium, 4 = hard, 5 = inter, 6 = wet

		uint32 m_inPits = 13;           // 0 = none, 1 = pitting, 2 = in pit area

		uint32 m_sector = 14;           // 0 = sector1, 1 = sector2, 2 = sector3

		uint32 m_currentLapInvalid = 15; // current lap invalid - 0 = valid, 1 = invalid

		uint32 m_penalties = 16;  // NEW: accumulated time penalties in seconds to be added
}
*/
deepf1::protobuf::CarUDPData UDPStreamUtils::toProto(const deepf1::CarUDPData& fromStream)
{
    deepf1::protobuf::CarUDPData rtn;
    for(unsigned int i = 0 ; i <=3 ; i++)
    {
        rtn.add_m_worldposition(fromStream.m_worldPosition[i]);
    }

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
/*
message UDPData {

}
*/
deepf1::protobuf::UDPData UDPStreamUtils::toProto(const deepf1::UDPPacket& fromStream)
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

	rtn.set_m_time(fromStream.m_lapTime);

	for(unsigned int i = 0 ; i < 4 ; i++)
    {
		rtn.add_m_wheel_speed(fromStream.m_wheel_speed[i]);
		rtn.add_m_susp_vel(fromStream.m_susp_vel[i]);
		rtn.add_m_susp_pos(fromStream.m_susp_pos[i]);
		rtn.add_m_susp_acceleration(fromStream.m_susp_acceleration[i]);
        rtn.add_m_tyres_damage(fromStream.m_tyres_damage[i]);
        rtn.add_m_tyres_wear(fromStream.m_tyres_wear[i]);
        rtn.add_m_tyres_temperature(fromStream.m_tyres_temperature[i]);
        rtn.add_m_tyres_pressure(fromStream.m_tyres_pressure[i]);
        rtn.add_m_brakes_temp(fromStream.m_brakes_temp[i]);
    }

    for(unsigned int i = 0 ; i < 20 ; i++)
    {
        *(rtn.add_m_car_data()) = toProto(fromStream.m_car_data[i]);
    }

    return rtn;
}

}