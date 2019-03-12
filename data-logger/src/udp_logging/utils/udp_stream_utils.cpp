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
		float m_time = 1;

		float m_lapTime = 2;

		float m_lapDistance = 3;

		float m_totalDistance = 4;

		float m_x = 5; // World space position

		float m_y = 6; // World space position

		float m_z = 7; // World space position

		float m_speed = 8; // Speed of car in MPH

		float m_xv = 9; // Velocity in world space

		float m_yv = 10; // Velocity in world space

		float m_zv = 11; // Velocity in world space

		float m_xr = 12; // World space right direction

		float m_yr = 13; // World space right direction

		float m_zr = 14; // World space right direction

		float m_xd = 15; // World space forward direction

		float m_yd = 16; // World space forward direction

		float m_zd = 17; // World space forward direction

		repeated float m_susp_pos = 18; // Note: All wheel arrays have the order:

		repeated float m_susp_vel = 19; // RL, RR, FL, FR

		repeated float m_wheel_speed = 20;

		float m_throttle = 21;

		float m_steer = 22;

		float m_brake = 23;

		float m_clutch = 24;

		float m_gear = 25;

		float m_gforce_lat = 26;

		float m_gforce_lon = 27;

		float m_lap = 28;

		float m_engineRate = 29;

		float m_sli_pro_native_support = 30; // SLI Pro support

		float m_car_position = 31; // car race position

		float m_kers_level = 32; // kers energy left

		float m_kers_max_level = 33; // kers maximum energy

		float m_drs = 34; // 0 = off, 1 = on

		float m_traction_control = 35; // 0 (off) - 2 (high)

		float m_anti_lock_brakes = 36; // 0 (off) - 1 (on)

		float m_fuel_in_tank = 37; // current fuel mass

		float m_fuel_capacity = 38; // fuel capacity

		float m_in_pits = 39; // 0 = none, 1 = pitting, 2 = in pit area

		float m_sector = 40; // 0 = sector1, 1 = sector2, 2 = sector3

		float m_sector1_time = 41; // time of sector1 (or 0)

		float m_sector2_time = 42; // time of sector2 (or 0)

		repeated float m_brakes_temp = 43; // brakes temperature (centigrade)

		repeated float m_tyres_pressure= 44; // tyres pressure PSI

		float m_team_info = 45; // team ID 

		float m_total_laps = 46; // total number of laps in this race

		float m_track_size = 47; // track size meters

		float m_last_lap_time = 48; // last lap time

		float m_max_rpm = 49; // cars max RPM, at which point the rev limiter will kick in

		float m_idle_rpm = 50; // cars idle RPM

		float m_max_gears = 51; // maximum number of gears

		float m_sessionType = 52; // 0 = unknown, 1 = practice, 2 = qualifying, 3 = race

		



}
*/
deepf1::protobuf::UDPData UDPStreamUtils::toProto(const deepf1::UDPPacket& fromStream)
{
    deepf1::protobuf::UDPData rtn;
    rtn.set_m_ang_acc_z(fromStream.m_ang_acc_z);
    rtn.set_m_ang_acc_y(fromStream.m_ang_acc_y);
    rtn.set_m_ang_acc_x(fromStream.m_ang_acc_x);

    for(unsigned int i = 0 ; i < 4 ; i++)
    {
        rtn.add_m_susp_acceleration(fromStream.m_susp_acceleration[i]);
    }

    rtn.set_m_z_local_velocity(fromStream.m_z_local_velocity);
    rtn.set_m_y_local_velocity(fromStream.m_y_local_velocity);
    rtn.set_m_x_local_velocity(fromStream.m_x_local_velocity);

    rtn.set_m_roll(fromStream.m_roll);
    rtn.set_m_pitch(fromStream.m_pitch);
    rtn.set_m_yaw(fromStream.m_yaw);

    for(unsigned int i = 0 ; i < 20 ; i++)
    {
        *(rtn.add_m_car_data()) = toProto(fromStream.m_car_data[i]);
    }
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

	
	for(unsigned int i = 0 ; i < 4 ; i++)
    {
    }


	rtn.set_m_currentlapinvalid(fromStream.m_currentLapInvalid);

	rtn.set_m_fuel_mix(fromStream.m_fuel_mix);

	rtn.set_m_front_brake_bias(fromStream.m_front_brake_bias);

	rtn.set_m_tyre_compound(fromStream.m_tyre_compound);

	rtn.set_m_ang_vel_z(fromStream.m_ang_vel_z);

	for(unsigned int i = 0 ; i < 4 ; i++)
    {
        rtn.add_m_tyres_damage(fromStream.m_tyres_damage[i]);
        rtn.add_m_tyres_wear(fromStream.m_tyres_wear[i]);
        rtn.add_m_tyres_temperature(fromStream.m_tyres_temperature[i]);
    }


/*
float m_drsAllowed = 53; // 0 = not allowed, 1 = allowed, -1 = invalid / unknown

		float m_track_number = 54; // -1 for unknown, 0-21 for tracks

		float m_vehicleFIAFlags = 55; // -1 = invalid/unknown, 0 = none, 1 = green, 2 = blue, 3 = yellow, 4 = red

		float m_era = 56;                     // era, 2017 (modern) or 1980 (classic)

		float m_engine_temperature = 57;   // engine temperature (centigrade)

		float m_gforce_vert = 58; // vertical g-force component
*/

	rtn.set_m_ang_vel_y(fromStream.m_ang_vel_y);

	rtn.set_m_ang_vel_x(fromStream.m_ang_vel_x);
	
	rtn.set_m_gforce_vert(fromStream.m_gforce_vert);

	rtn.set_m_engine_temperature(fromStream.m_engine_temperature);

    return rtn;
}

}