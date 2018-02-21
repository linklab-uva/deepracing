#include "deepf1_gsoap_conversions/gsoap_conversions.h"


namespace deepf1_gsoap_conversions {

	
	gsoap_conversions::~gsoap_conversions()
	{
	}
	deepf1_gsoap::CarUDPData gsoap_conversions::convert_to_gsoap(const deepf1::CarUDPData& car_data) {
		deepf1_gsoap::CarUDPData rtn;
		rtn.m_bestLapTime = car_data.m_bestLapTime;
		rtn.m_carPosition = car_data.m_carPosition;
		rtn.m_currentLapInvalid = car_data.m_currentLapInvalid;
		rtn.m_currentLapNum = car_data.m_currentLapNum;
		rtn.m_currentLapTime = car_data.m_currentLapTime;
		rtn.m_driverId = car_data.m_driverId;
		rtn.m_inPits = car_data.m_inPits;
		rtn.m_lapDistance = car_data.m_lapDistance;
		rtn.m_lastLapTime = car_data.m_lastLapTime;
		rtn.m_penalties = car_data.m_penalties;
		rtn.m_sector = car_data.m_sector;
		rtn.m_sector1Time = car_data.m_sector1Time;
		rtn.m_sector2Time = car_data.m_sector2Time;
		rtn.m_teamId = car_data.m_teamId;
		rtn.m_tyreCompound = car_data.m_tyreCompound;
		for (unsigned int i = 0; i < 3; i++) {
			rtn.m_worldPosition[i] = car_data.m_worldPosition[i];
		}
		return rtn;
	}
	deepf1_gsoap::CarUDPData* gsoap_conversions::convert_to_gsoap_dynamic(const deepf1::CarUDPData& car_data) {
		deepf1_gsoap::CarUDPData* rtn = deepf1_gsoap::soap_new_CarUDPData(soap);
		rtn->m_bestLapTime = car_data.m_bestLapTime;
		rtn->m_carPosition = car_data.m_carPosition;
		rtn->m_currentLapInvalid = car_data.m_currentLapInvalid;
		rtn->m_currentLapNum = car_data.m_currentLapNum;
		rtn->m_currentLapTime = car_data.m_currentLapTime;
		rtn->m_driverId = car_data.m_driverId;
		rtn->m_inPits = car_data.m_inPits;
		rtn->m_lapDistance = car_data.m_lapDistance;
		rtn->m_lastLapTime = car_data.m_lastLapTime;
		rtn->m_penalties = car_data.m_penalties;
		rtn->m_sector = car_data.m_sector;
		rtn->m_sector1Time = car_data.m_sector1Time;
		rtn->m_sector2Time = car_data.m_sector2Time;
		rtn->m_teamId = car_data.m_teamId;
		rtn->m_tyreCompound = car_data.m_tyreCompound;
		for (unsigned int i = 0; i < 3; i++) {
			rtn->m_worldPosition[i] = car_data.m_worldPosition[i];
		}
		return rtn;
	}
	deepf1_gsoap::UDPPacket* gsoap_conversions::convert_to_gsoap(const deepf1::UDPPacket& udp_data) {
		deepf1_gsoap::UDPPacket* rtn = deepf1_gsoap::soap_new_UDPPacket(soap);
		rtn->m_ang_acc_x = udp_data.m_ang_acc_x;
		rtn->m_ang_acc_y = udp_data.m_ang_acc_y;
		rtn->m_ang_acc_z = udp_data.m_ang_acc_z;
		rtn->m_ang_vel_x = udp_data.m_ang_vel_x;
		rtn->m_ang_vel_y = udp_data.m_ang_vel_y;
		rtn->m_ang_vel_z = udp_data.m_ang_vel_z;
		rtn->m_anti_lock_brakes = udp_data.m_anti_lock_brakes;
		rtn->m_brake = udp_data.m_brake;
		rtn->m_car_position = udp_data.m_car_position;
		rtn->m_clutch = udp_data.m_clutch;
		rtn->m_currentLapInvalid = udp_data.m_currentLapInvalid;
		rtn->m_drs = udp_data.m_drs;
		rtn->m_drsAllowed = udp_data.m_drsAllowed;
		rtn->m_engineRate = udp_data.m_engineRate;
		rtn->m_engine_damage = udp_data.m_engine_damage;
		rtn->m_engine_temperature = udp_data.m_engine_temperature;
		rtn->m_era = udp_data.m_era;
		rtn->m_exhaust_damage = udp_data.m_exhaust_damage;
		rtn->m_front_brake_bias = udp_data.m_front_brake_bias;
		rtn->m_front_left_wing_damage = udp_data.m_front_left_wing_damage;
		rtn->m_front_right_wing_damage = udp_data.m_front_right_wing_damage;
		rtn->m_fuel_capacity = udp_data.m_fuel_capacity;
		rtn->m_fuel_in_tank = udp_data.m_fuel_in_tank;
		rtn->m_fuel_mix = udp_data.m_fuel_mix;
		rtn->m_gear = udp_data.m_gear;
		rtn->m_gear_box_damage = udp_data.m_gear_box_damage;
		rtn->m_gforce_lat = udp_data.m_gforce_lat;
		rtn->m_gforce_lon = udp_data.m_gforce_lon;
		rtn->m_gforce_vert = udp_data.m_gforce_vert;
		rtn->m_idle_rpm = udp_data.m_idle_rpm;
		rtn->m_in_pits = udp_data.m_in_pits;
		rtn->m_is_spectating = udp_data.m_is_spectating;
		rtn->m_kers_level = udp_data.m_kers_level;
		rtn->m_kers_max_level = udp_data.m_kers_max_level;
		rtn->m_lap = udp_data.m_lap;
		rtn->m_lapDistance = udp_data.m_lapDistance;
		rtn->m_lapTime = udp_data.m_lapTime;
		rtn->m_last_lap_time = udp_data.m_last_lap_time;
		rtn->m_max_gears = udp_data.m_max_gears;
		rtn->m_max_rpm = udp_data.m_max_rpm;
		rtn->m_num_cars = udp_data.m_num_cars;
		rtn->m_pitch = udp_data.m_pitch;
		rtn->m_pit_limiter_status = udp_data.m_pit_limiter_status;
		rtn->m_pit_speed_limit = udp_data.m_pit_speed_limit;
		rtn->m_player_car_index = udp_data.m_player_car_index;
		rtn->m_rear_wing_damage = udp_data.m_rear_wing_damage;
		rtn->m_rev_lights_percent = udp_data.m_rev_lights_percent;
		rtn->m_roll = udp_data.m_roll;
		rtn->m_sector = udp_data.m_sector;
		rtn->m_sector1_time = udp_data.m_sector1_time;
		rtn->m_sector2_time = udp_data.m_sector2_time;
		rtn->m_sessionType = udp_data.m_sessionType;
		rtn->m_session_time_left = udp_data.m_session_time_left;
		rtn->m_sli_pro_native_support = udp_data.m_sli_pro_native_support;
		rtn->m_spectator_car_index = udp_data.m_spectator_car_index;
		rtn->m_speed = udp_data.m_speed;
		rtn->m_steer = udp_data.m_steer;
		rtn->m_team_info = udp_data.m_team_info;
		rtn->m_throttle = udp_data.m_throttle;
		rtn->m_time = udp_data.m_time;
		rtn->m_totalDistance = udp_data.m_totalDistance;
		rtn->m_total_laps = udp_data.m_total_laps;
		rtn->m_track_number = udp_data.m_track_number;
		rtn->m_track_size = udp_data.m_track_size;
		rtn->m_traction_control = udp_data.m_traction_control;
		rtn->m_tyre_compound = udp_data.m_tyre_compound;
		rtn->m_vehicleFIAFlags = udp_data.m_vehicleFIAFlags;
		rtn->m_x = udp_data.m_x;
		rtn->m_xd = udp_data.m_xd;
		rtn->m_xr = udp_data.m_xr;
		rtn->m_xv = udp_data.m_xv;
		rtn->m_x_local_velocity = udp_data.m_x_local_velocity;
		rtn->m_y = udp_data.m_y;
		rtn->m_yd = udp_data.m_yd;
		rtn->m_yr = udp_data.m_yr;
		rtn->m_yv = udp_data.m_yv;
		rtn->m_y_local_velocity = udp_data.m_y_local_velocity;
		rtn->m_yaw = udp_data.m_yaw;
		rtn->m_z = udp_data.m_z;
		rtn->m_zd = udp_data.m_zd;
		rtn->m_zr = udp_data.m_zr;
		rtn->m_zv = udp_data.m_zv;
		rtn->m_z_local_velocity = udp_data.m_z_local_velocity;
		for (unsigned int i = 0; i < 20; i++) {
			deepf1_gsoap::CarUDPData* gsoap_car_data = convert_to_gsoap_dynamic(udp_data.m_car_data[i]);
			rtn->m_car_data[i] = *gsoap_car_data;
		}
		for (unsigned int i = 0; i < 4; i++) {
			rtn->m_brakes_temp[i] = udp_data.m_brakes_temp[i];
			rtn->m_susp_acceleration[i] = udp_data.m_susp_acceleration[i];
			rtn->m_susp_pos[i] = udp_data.m_susp_pos[i];
			rtn->m_susp_vel[i] = udp_data.m_susp_vel[i];
			rtn->m_tyres_damage[i] = udp_data.m_tyres_damage[i];
			rtn->m_tyres_pressure[i] = udp_data.m_tyres_pressure[i];
			rtn->m_tyres_temperature[i] = udp_data.m_tyres_temperature[i];
			rtn->m_tyres_wear[i] = udp_data.m_tyres_wear[i];
			rtn->m_wheel_speed[i] = udp_data.m_wheel_speed[i];
		}
		return rtn;
	}
}