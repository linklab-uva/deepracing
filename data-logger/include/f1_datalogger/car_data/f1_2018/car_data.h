



#ifndef INCLUDE_F1_2018_CAR_DATA_H
#define INCLUDE_F1_2018_CAR_DATA_H
#include <f1_datalogger/car_data/typedefs.h>

namespace deepf1{
namespace twenty_eighteen{

	enum PacketID
	{
		MOTION=0,
		SESSION=1,
		LAPDATA=2,
		EVENT=3,
		PARTICIPANTS=4,
		CARSETUPS=5,
		CARTELEMETRY=6,
		CARSTATUS=7
	};

	PACK(
	struct PacketHeader
	{
		uint16    m_packetFormat;         // 2018
		uint8      m_packetVersion;        // Version of this packet type, all start from 1
		uint8      m_packetId;             // Identifier for the packet type, see below
		uint64     m_sessionUID;           // Unique identifier for the session
		float       m_sessionTime;          // Session timestamp
		uint     m_frameIdentifier;      // Identifier for the frame the data was retrieved on
		uint8     m_playerCarIndex;       // Index of player's car in the array
	}
	);
	PACK(
	struct CarMotionData
	{
		float         m_worldPositionX;           // World space X position
		float         m_worldPositionY;           // World space Y position
		float         m_worldPositionZ;           // World space Z position
		float         m_worldVelocityX;           // Velocity in world space X
		float         m_worldVelocityY;           // Velocity in world space Y
		float         m_worldVelocityZ;           // Velocity in world space Z
		int16_t       m_worldForwardDirX;         // World space forward X direction (normalised)
		int16_t       m_worldForwardDirY;         // World space forward Y direction (normalised)
		int16_t       m_worldForwardDirZ;         // World space forward Z direction (normalised)
		int16_t       m_worldRightDirX;           // World space right X direction (normalised)
		int16_t       m_worldRightDirY;           // World space right Y direction (normalised)
		int16_t       m_worldRightDirZ;           // World space right Z direction (normalised)
		float         m_gForceLateral;            // Lateral G-Force component
		float         m_gForceLongitudinal;       // Longitudinal G-Force component
		float         m_gForceVertical;           // Vertical G-Force component
		float         m_yaw;                      // Yaw angle in radians
		float         m_pitch;                    // Pitch angle in radians
		float         m_roll;                     // Roll angle in radians
	}
	);
	PACK(
	struct PacketMotionData
	{
		PacketHeader    m_header;               // Header

		CarMotionData   m_carMotionData[20];    // Data for all cars on track

		// Extra player car ONLY data
		float         m_suspensionPosition[4];       // Note: All wheel arrays have the following order:
		float         m_suspensionVelocity[4];       // RL, RR, FL, FR
		float         m_suspensionAcceleration[4];   // RL, RR, FL, FR
		float         m_wheelSpeed[4];               // Speed of each wheel
		float         m_wheelSlip[4];                // Slip ratio for each wheel
		float         m_localVelocityX;              // Velocity in local space
		float         m_localVelocityY;              // Velocity in local space
		float         m_localVelocityZ;              // Velocity in local space
		float         m_angularVelocityX;            // Angular velocity x-component
		float         m_angularVelocityY;            // Angular velocity y-component
		float         m_angularVelocityZ;            // Angular velocity z-component
		float         m_angularAccelerationX;        // Angular velocity x-component
		float         m_angularAccelerationY;        // Angular velocity y-component
		float         m_angularAccelerationZ;        // Angular velocity z-component
		float         m_frontWheelsAngle;            // Current front wheels angle in radians
	}
	);
	PACK(
	struct MarshalZone
	{
		float  m_zoneStart;   // Fraction (0..1) of way through the lap the marshal zone starts
		int8_t   m_zoneFlag;    // -1 = invalid/unknown, 0 = none, 1 = green, 2 = blue, 3 = yellow, 4 = red
	}
	);
	PACK(
	struct PacketSessionData
	{
		PacketHeader    m_header;               	// Header

		uint8_t         m_weather;              	// Weather - 0 = clear, 1 = light cloud, 2 = overcast
													// 3 = light rain, 4 = heavy rain, 5 = storm
		int8_t			m_trackTemperature;    	// Track temp. in degrees celsius
		int8_t			m_airTemperature;      	// Air temp. in degrees celsius
		uint8_t         m_totalLaps;           	// Total number of laps in this race
		uint16_t        m_trackLength;           	// Track length in metres
		uint8_t         m_sessionType;         	// 0 = unknown, 1 = P1, 2 = P2, 3 = P3, 4 = Short P
													// 5 = Q1, 6 = Q2, 7 = Q3, 8 = Short Q, 9 = OSQ
													// 10 = R, 11 = R2, 12 = Time Trial
		int8_t          m_trackId;         		// -1 for unknown, 0-21 for tracks, see appendix
		uint8_t         m_era;                  	// Era, 0 = modern, 1 = classic
		uint16_t        m_sessionTimeLeft;    	// Time left in session in seconds
		uint16_t        m_sessionDuration;     	// Session duration in seconds
		uint8_t         m_pitSpeedLimit;      	// Pit speed limit in kilometres per hour
		uint8_t         m_gamePaused;               // Whether the game is paused
		uint8_t         m_isSpectating;        	// Whether the player is spectating
		uint8_t         m_spectatorCarIndex;  	// Index of the car being spectated
		uint8_t         m_sliProNativeSupport;	// SLI Pro support, 0 = inactive, 1 = active
		uint8_t         m_numMarshalZones;         	// Number of marshal zones to follow
		MarshalZone     m_marshalZones[21];         // List of marshal zones – max 21
		uint8_t         m_safetyCarStatus;          // 0 = no safety car, 1 = full safety car
													// 2 = virtual safety car
		uint8_t         m_networkGame;              // 0 = offline, 1 = online
	}
	);

	PACK(
	struct LapData
	{
		float       m_lastLapTime;           // Last lap time in seconds
		float       m_currentLapTime;        // Current time around the lap in seconds
		float       m_bestLapTime;           // Best lap time of the session in seconds
		float       m_sector1Time;           // Sector 1 time in seconds
		float       m_sector2Time;           // Sector 2 time in seconds
		float       m_lapDistance;           // Distance vehicle is around current lap in metres – could
											 // be negative if line hasn’t been crossed yet
		float       m_totalDistance;         // Total distance travelled in session in metres – could
											 // be negative if line hasn’t been crossed yet
		float       m_safetyCarDelta;        // Delta in seconds for safety car
		uint8_t       m_carPosition;           // Car race position
		uint8_t       m_currentLapNum;         // Current lap number
		uint8_t       m_pitStatus;             // 0 = none, 1 = pitting, 2 = in pit area
		uint8_t       m_sector;                // 0 = sector1, 1 = sector2, 2 = sector3
		uint8_t       m_currentLapInvalid;     // Current lap invalid - 0 = valid, 1 = invalid
		uint8_t       m_penalties;             // Accumulated time penalties in seconds to be added
		uint8_t       m_gridPosition;          // Grid position the vehicle started the race in
		uint8_t       m_driverStatus;          // Status of driver - 0 = in garage, 1 = flying lap
											 // 2 = in lap, 3 = out lap, 4 = on track
		uint8_t       m_resultStatus;          // Result status - 0 = invalid, 1 = inactive, 2 = active
											 // 3 = finished, 4 = disqualified, 5 = not classified
											 // 6 = retired
	}
	);
	PACK(
	struct PacketLapData
	{
		PacketHeader    m_header;              // Header

		LapData         m_lapData[20];         // Lap data for all cars on track
	}
	);
	PACK(
	struct PacketEventData
	{
		PacketHeader    m_header;               // Header

		uint8_t           m_eventStringCode[4];   // Event string code, see above
	}
	);
	PACK(
	struct ParticipantData
	{
		uint8_t      m_aiControlled;           // Whether the vehicle is AI (1) or Human (0) controlled
		uint8_t     m_driverId;               // Driver id - see appendix
		uint8_t      m_teamId;                 // Team id - see appendix
		uint8_t      m_raceNumber;             // Race number of the car
		uint8_t      m_nationality;            // Nationality of the driver
		char       m_name[48];               // Name of participant in UTF-8 format – null terminated
											 // Will be truncated with … (U+2026) if too long
	}
	);
	PACK(
	struct PacketParticipantsData
	{
		PacketHeader    m_header;            // Header

		uint8_t           m_numCars;           // Number of cars in the data
		ParticipantData m_participants[20];
	}
	);

	PACK(
	struct CarSetupData
	{
		uint8_t     m_frontWing;                // Front wing aero
		uint8_t     m_rearWing;                 // Rear wing aero
		uint8_t     m_onThrottle;               // Differential adjustment on throttle (percentage)
		uint8_t     m_offThrottle;              // Differential adjustment off throttle (percentage)
		float     m_frontCamber;              // Front camber angle (suspension geometry)
		float     m_rearCamber;               // Rear camber angle (suspension geometry)
		float     m_frontToe;                 // Front toe angle (suspension geometry)
		float     m_rearToe;                  // Rear toe angle (suspension geometry)
		uint8_t     m_frontSuspension;          // Front suspension
		uint8_t     m_rearSuspension;           // Rear suspension
		uint8_t     m_frontAntiRollBar;         // Front anti-roll bar
		uint8_t     m_rearAntiRollBar;          // Front anti-roll bar
		uint8_t     m_frontSuspensionHeight;    // Front ride height
		uint8_t     m_rearSuspensionHeight;     // Rear ride height
		uint8_t     m_brakePressure;            // Brake pressure (percentage)
		uint8_t     m_brakeBias;                // Brake bias (percentage)
		float     m_frontTyrePressure;        // Front tyre pressure (PSI)
		float     m_rearTyrePressure;         // Rear tyre pressure (PSI)
		uint8_t     m_ballast;                  // Ballast
		float     m_fuelLoad;                 // Fuel load
	}
	);
	PACK(
	struct PacketCarSetupData
	{
		PacketHeader    m_header;            // Header

		CarSetupData    m_carSetups[20];
	}
	);
	PACK(
	struct CarTelemetryData
	{
		uint16_t  m_speed;                      // Speed of car in kilometres per hour
		uint8_t   m_throttle;                   // Amount of throttle applied (0 to 100)
		int8_t    m_steer;                      // Steering (-100 (full lock left) to 100 (full lock right))
		uint8_t   m_brake;                      // Amount of brake applied (0 to 100)
		uint8_t   m_clutch;                     // Amount of clutch applied (0 to 100)
		int8_t    m_gear;                       // Gear selected (1-8, N=0, R=-1)
		uint16_t  m_engineRPM;                  // Engine RPM
		uint8_t   m_drs;                        // 0 = off, 1 = on
		uint8_t   m_revLightsPercent;           // Rev lights indicator (percentage)
		uint16_t  m_brakesTemperature[4];       // Brakes temperature (celsius)
		uint16_t  m_tyresSurfaceTemperature[4]; // Tyres surface temperature (celsius)
		uint16_t  m_tyresInnerTemperature[4];   // Tyres inner temperature (celsius)
		uint16_t  m_engineTemperature;          // Engine temperature (celsius)
		float     m_tyresPressure[4];           // Tyres pressure (PSI)
	}
	);
	PACK(
	struct PacketCarTelemetryData
	{
		PacketHeader        m_header;                // Header

		CarTelemetryData    m_carTelemetryData[20];

		uint32_t              m_buttonStatus;         // Bit flags specifying which buttons are being
													// pressed currently - see appendices
	}
	);
	PACK(
	struct CarStatusData
	{
		uint8_t       m_tractionControl;          // 0 (off) - 2 (high)
		uint8_t       m_antiLockBrakes;           // 0 (off) - 1 (on)
		uint8_t       m_fuelMix;                  // Fuel mix - 0 = lean, 1 = standard, 2 = rich, 3 = max
		uint8_t       m_frontBrakeBias;           // Front brake bias (percentage)
		uint8_t       m_pitLimiterStatus;         // Pit limiter status - 0 = off, 1 = on
		float       m_fuelInTank;               // Current fuel mass
		float       m_fuelCapacity;             // Fuel capacity
		uint16_t      m_maxRPM;                   // Cars max RPM, point of rev limiter
		uint16_t      m_idleRPM;                  // Cars idle RPM
		uint8_t       m_maxGears;                 // Maximum number of gears
		uint8_t       m_drsAllowed;               // 0 = not allowed, 1 = allowed, -1 = unknown
		uint8_t       m_tyresWear[4];             // Tyre wear percentage
		uint8_t       m_tyreCompound;             // Modern - 0 = hyper soft, 1 = ultra soft
												// 2 = super soft, 3 = soft, 4 = medium, 5 = hard
												// 6 = super hard, 7 = inter, 8 = wet
												// Classic - 0-6 = dry, 7-8 = wet
		uint8_t       m_tyresDamage[4];           // Tyre damage (percentage)
		uint8_t       m_frontLeftWingDamage;      // Front left wing damage (percentage)
		uint8_t       m_frontRightWingDamage;     // Front right wing damage (percentage)
		uint8_t       m_rearWingDamage;           // Rear wing damage (percentage)
		uint8_t       m_engineDamage;             // Engine damage (percentage)
		uint8_t       m_gearBoxDamage;            // Gear box damage (percentage)
		uint8_t       m_exhaustDamage;            // Exhaust damage (percentage)
		int8_t        m_vehicleFiaFlags;          // -1 = invalid/unknown, 0 = none, 1 = green
												// 2 = blue, 3 = yellow, 4 = red
		float       m_ersStoreEnergy;           // ERS energy store in Joules
		uint8_t       m_ersDeployMode;            // ERS deployment mode, 0 = none, 1 = low, 2 = medium
												// 3 = high, 4 = overtake, 5 = hotlap
		float       m_ersHarvestedThisLapMGUK;  // ERS energy harvested this lap by MGU-K
		float       m_ersHarvestedThisLapMGUH;  // ERS energy harvested this lap by MGU-H
		float       m_ersDeployedThisLap;       // ERS energy deployed this lap
	}
	);
	PACK(
	struct PacketCarStatusData
	{
		PacketHeader        m_header;            // Header

		CarStatusData       m_carStatusData[20];
	}
	);
}

	PACK(
	struct CarUDPData2017

	{

		float m_worldPosition[3]; // world co-ordinates of vehicle

		float m_lastLapTime;

		float m_currentLapTime;

		float m_bestLapTime;

		float m_sector1Time;

		float m_sector2Time;

		float m_lapDistance;

		char  m_driverId;

		char  m_teamId;

		char  m_carPosition;     // UPDATED: track positions of vehicle

		char  m_currentLapNum;

		char  m_tyreCompound; // compound of tyre � 0 = ultra soft, 1 = super soft, 2 = soft, 3 = medium, 4 = hard, 5 = inter, 6 = wet

		char  m_inPits;           // 0 = none, 1 = pitting, 2 = in pit area

		char  m_sector;           // 0 = sector1, 1 = sector2, 2 = sector3

		char  m_currentLapInvalid; // current lap invalid - 0 = valid, 1 = invalid

		char  m_penalties;  // NEW: accumulated time penalties in seconds to be added
	}); 

	PACK(
	struct UDPPacket2017
	{

		float m_time;

		float m_lapTime;

		float m_lapDistance;

		float m_totalDistance;

		float m_x; // World space position

		float m_y; // World space position

		float m_z; // World space position

		float m_speed; // Speed of car in MPH

		float m_xv; // Velocity in world space

		float m_yv; // Velocity in world space

		float m_zv; // Velocity in world space

		float m_xr; // World space right direction

		float m_yr; // World space right direction

		float m_zr; // World space right direction

		float m_xd; // World space forward direction

		float m_yd; // World space forward direction

		float m_zd; // World space forward direction

		float m_susp_pos[4]; // Note: All wheel arrays have the order:

		float m_susp_vel[4]; // RL, RR, FL, FR

		float m_wheel_speed[4];

		float m_throttle;

		float m_steer;

		float m_brake;

		float m_clutch;

		float m_gear;

		float m_gforce_lat;

		float m_gforce_lon;

		float m_lap;

		float m_engineRate;

		float m_sli_pro_native_support; // SLI Pro support

		float m_car_position; // car race position

		float m_kers_level; // kers energy left

		float m_kers_max_level; // kers maximum energy

		float m_drs; // 0 = off, 1 = on

		float m_traction_control; // 0 (off) - 2 (high)

		float m_anti_lock_brakes; // 0 (off) - 1 (on)

		float m_fuel_in_tank; // current fuel mass

		float m_fuel_capacity; // fuel capacity

		float m_in_pits; // 0 = none, 1 = pitting, 2 = in pit area

		float m_sector; // 0 = sector1, 1 = sector2, 2 = sector3

		float m_sector1_time; // time of sector1 (or 0)

		float m_sector2_time; // time of sector2 (or 0)

		float m_brakes_temp[4]; // brakes temperature (centigrade)

		float m_tyres_pressure[4]; // tyres pressure PSI

		float m_team_info; // team ID 

		float m_total_laps; // total number of laps in this race

		float m_track_size; // track size meters

		float m_last_lap_time; // last lap time

		float m_max_rpm; // cars max RPM, at which point the rev limiter will kick in

		float m_idle_rpm; // cars idle RPM

		float m_max_gears; // maximum number of gears

		float m_sessionType; // 0 = unknown, 1 = practice, 2 = qualifying, 3 = race

		float m_drsAllowed; // 0 = not allowed, 1 = allowed, -1 = invalid / unknown

		float m_track_number; // -1 for unknown, 0-21 for tracks

		float m_vehicleFIAFlags; // -1 = invalid/unknown, 0 = none, 1 = green, 2 = blue, 3 = yellow, 4 = red

		float m_era;                     // era, 2017 (modern) or 1980 (classic)

		float m_engine_temperature;   // engine temperature (centigrade)

		float m_gforce_vert; // vertical g-force component

		float m_ang_vel_x; // angular velocity x-component

		float m_ang_vel_y; // angular velocity y-component

		float m_ang_vel_z; // angular velocity z-component

		char  m_tyres_temperature[4]; // tyres temperature (centigrade)

		char  m_tyres_wear[4]; // tyre wear percentage

		char  m_tyre_compound; // compound of tyre � 0 = ultra soft, 1 = super soft, 2 = soft, 3 = medium, 4 = hard, 5 = inter, 6 = wet

		char  m_front_brake_bias;         // front brake bias (percentage)

		char  m_fuel_mix;                 // fuel mix - 0 = lean, 1 = standard, 2 = rich, 3 = max

		char  m_currentLapInvalid;     // current lap invalid - 0 = valid, 1 = invalid

		char  m_tyres_damage[4]; // tyre damage (percentage)

		char  m_front_left_wing_damage; // front left wing damage (percentage)

		char  m_front_right_wing_damage; // front right wing damage (percentage)

		char  m_rear_wing_damage; // rear wing damage (percentage)

		char  m_engine_damage; // engine damage (percentage)

		char  m_gear_box_damage; // gear box damage (percentage)

		char  m_exhaust_damage; // exhaust damage (percentage)

		char  m_pit_limiter_status; // pit limiter status � 0 = off, 1 = on

		char  m_pit_speed_limit; // pit speed limit in mph

		float m_session_time_left;  // NEW: time left in session in seconds 

		char  m_rev_lights_percent;  // NEW: rev lights indicator (percentage)

		char  m_is_spectating;  // NEW: whether the player is spectating

		char  m_spectator_car_index;  // NEW: index of the car being spectated


									  // Car data

		char  m_num_cars;               // number of cars in data

		char  m_player_car_index;         // index of player's car in the array

		CarUDPData2017  m_car_data[20];   // data for all cars on track


		float m_yaw;  // NEW (v1.8)

		float m_pitch;  // NEW (v1.8)

		float m_roll;  // NEW (v1.8)

		float m_x_local_velocity;          // NEW (v1.8) Velocity in local space

		float m_y_local_velocity;          // NEW (v1.8) Velocity in local space

		float m_z_local_velocity;          // NEW (v1.8) Velocity in local space

		float m_susp_acceleration[4];   // NEW (v1.8) RL, RR, FL, FR

		float m_ang_acc_x;                 // NEW (v1.8) angular acceleration x-component

		float m_ang_acc_y;                 // NEW (v1.8) angular acceleration y-component

		float m_ang_acc_z;                 // NEW (v1.8) angular acceleration z-component

	});
}
#endif
