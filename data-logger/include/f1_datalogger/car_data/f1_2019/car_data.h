



#ifndef INCLUDE_F1_2019_CAR_DATA_H
#define INCLUDE_F1_2019_CAR_DATA_H

#include <f1_datalogger/car_data/typedefs.h>

namespace deepf1{
namespace twenty_nineteen{

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
			uint16    m_packetFormat;         // 2019
			uint8     m_gameMajorVersion;     // Game major version - "X.00"
			uint8     m_gameMinorVersion;     // Game minor version - "1.XX"
			uint8     m_packetVersion;        // Version of this packet type, all start from 1
			uint8     m_packetId;             // Identifier for the packet type, see below
			uint64    m_sessionUID;           // Unique identifier for the session
			float     m_sessionTime;          // Session timestamp
			uint      m_frameIdentifier;      // Identifier for the frame the data was retrieved on
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
			int16         m_worldForwardDirX;         // World space forward X direction (normalised)
			int16         m_worldForwardDirY;         // World space forward Y direction (normalised)
			int16         m_worldForwardDirZ;         // World space forward Z direction (normalised)
			int16         m_worldRightDirX;           // World space right X direction (normalised)
			int16         m_worldRightDirY;           // World space right Y direction (normalised)
			int16         m_worldRightDirZ;           // World space right Z direction (normalised)
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
			PacketHeader    m_header;               	// Header

			CarMotionData   m_carMotionData[20];    	// Data for all cars on track

			// Extra player car ONLY data
			float         m_suspensionPosition[4];       // Note: All wheel arrays have the following order:
			float         m_suspensionVelocity[4];       // RL, RR, FL, FR
			float         m_suspensionAcceleration[4];	// RL, RR, FL, FR
			float         m_wheelSpeed[4];           	// Speed of each wheel
			float         m_wheelSlip[4];                // Slip ratio for each wheel
			float         m_localVelocityX;         	// Velocity in local space
			float         m_localVelocityY;         	// Velocity in local space
			float         m_localVelocityZ;         	// Velocity in local space
			float         m_angularVelocityX;		// Angular velocity x-component
			float         m_angularVelocityY;            // Angular velocity y-component
			float         m_angularVelocityZ;            // Angular velocity z-component
			float         m_angularAccelerationX;        // Angular velocity x-component
			float         m_angularAccelerationY;	// Angular velocity y-component
			float         m_angularAccelerationZ;        // Angular velocity z-component
			float         m_frontWheelsAngle;            // Current front wheels angle in radians
		}
	);
	PACK(
		struct MarshalZone
		{
			float  m_zoneStart;   // Fraction (0..1) of way through the lap the marshal zone starts
			int8   m_zoneFlag;    // -1 = invalid/unknown, 0 = none, 1 = green, 2 = blue, 3 = yellow, 4 = red
		}
	);
	PACK(
		struct PacketSessionData
		{
			PacketHeader    m_header;               	// Header

			uint8           m_weather;              	// Weather - 0 = clear, 1 = light cloud, 2 = overcast
														// 3 = light rain, 4 = heavy rain, 5 = storm
			int8			m_trackTemperature;    	// Track temp. in degrees celsius
			int8			m_airTemperature;      	// Air temp. in degrees celsius
			uint8           m_totalLaps;           	// Total number of laps in this race
			uint16          m_trackLength;           	// Track length in metres
			uint8           m_sessionType;         	// 0 = unknown, 1 = P1, 2 = P2, 3 = P3, 4 = Short P
														// 5 = Q1, 6 = Q2, 7 = Q3, 8 = Short Q, 9 = OSQ
														// 10 = R, 11 = R2, 12 = Time Trial
			int8            m_trackId;         		// -1 for unknown, 0-21 for tracks, see appendix
			uint8           m_formula;                  	// Formula, 0 = F1 Modern, 1 = F1 Classic, 2 = F2,
		// 3 = F1 Generic
			uint16          m_sessionTimeLeft;    	// Time left in session in seconds
			uint16          m_sessionDuration;     	// Session duration in seconds
			uint8           m_pitSpeedLimit;      	// Pit speed limit in kilometres per hour
			uint8           m_gamePaused;                // Whether the game is paused
			uint8           m_isSpectating;        	// Whether the player is spectating
			uint8           m_spectatorCarIndex;  	// Index of the car being spectated
			uint8           m_sliProNativeSupport;	// SLI Pro support, 0 = inactive, 1 = active
			uint8           m_numMarshalZones;         	// Number of marshal zones to follow
			MarshalZone     m_marshalZones[21];         	// List of marshal zones – max 21
			uint8           m_safetyCarStatus;           // 0 = no safety car, 1 = full safety car
														// 2 = virtual safety car
			uint8           m_networkGame;               // 0 = offline, 1 = online
		}
	);

	PACK(
		struct LapData
		{
			float       m_lastLapTime;	       	// Last lap time in seconds
			float       m_currentLapTime;	// Current time around the lap in seconds
			float       m_bestLapTime;		// Best lap time of the session in seconds
			float       m_sector1Time;		// Sector 1 time in seconds
			float       m_sector2Time;		// Sector 2 time in seconds
			float       m_lapDistance;		// Distance vehicle is around current lap in metres – could
							// be negative if line hasn’t been crossed yet
			float       m_totalDistance;		// Total distance travelled in session in metres – could
							// be negative if line hasn’t been crossed yet
			float       m_safetyCarDelta;        // Delta in seconds for safety car
			uint8       m_carPosition;   	// Car race position
			uint8       m_currentLapNum;		// Current lap number
			uint8       m_pitStatus;            	// 0 = none, 1 = pitting, 2 = in pit area
			uint8       m_sector;               	// 0 = sector1, 1 = sector2, 2 = sector3
			uint8       m_currentLapInvalid;    	// Current lap invalid - 0 = valid, 1 = invalid
			uint8       m_penalties;            	// Accumulated time penalties in seconds to be added
			uint8       m_gridPosition;         	// Grid position the vehicle started the race in
			uint8       m_driverStatus;         	// Status of driver - 0 = in garage, 1 = flying lap
		// 2 = in lap, 3 = out lap, 4 = on track
			uint8       m_resultStatus;          // Result status - 0 = invalid, 1 = inactive, 2 = active
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
		union EventDataDetails
		{
			struct
			{
				uint8	vehicleIdx; // Vehicle index of car achieving fastest lap
				float	lapTime;    // Lap time is in seconds
			} FastestLap;

			struct
			{
				uint8   vehicleIdx; // Vehicle index of car retiring
			} Retirement;

			struct
			{
				uint8   vehicleIdx; // Vehicle index of team mate
			} TeamMateInPits;

			struct
			{
				uint8   vehicleIdx; // Vehicle index of the race winner
			} RaceWinner;
		}
	);
	PACK(
		struct PacketEventData
		{
			PacketHeader    	m_header;               // Header
			
			uint8           	m_eventStringCode[4];   // Event string code, see below
			EventDataDetails	m_eventDetails;         // Event details - should be interpreted differently for each type
		}
	);
	PACK(
		struct ParticipantData
		{
			uint8      m_aiControlled;           // Whether the vehicle is AI (1) or Human (0) controlled
			uint8      m_driverId;		// Driver id - see appendix
			uint8      m_teamId;                 // Team id - see appendix
			uint8      m_raceNumber;             // Race number of the car
			uint8      m_nationality;            // Nationality of the driver
			char       m_name[48];               // Name of participant in UTF-8 format – null terminated. Will be truncated with … (U+2026) if too long
			uint8      m_yourTelemetry;          // The player's UDP setting, 0 = restricted, 1 = public
		}
	);
	PACK(
		struct PacketParticipantsData
		{
			PacketHeader    m_header;            // Header

			uint8           m_numActiveCars;	// Number of active cars in the data – should match number of cars on HUD
			ParticipantData m_participants[20];
		}
	);

	PACK(
		struct CarSetupData
		{
			uint8     m_frontWing;                // Front wing aero
			uint8     m_rearWing;                 // Rear wing aero
			uint8     m_onThrottle;               // Differential adjustment on throttle (percentage)
			uint8     m_offThrottle;              // Differential adjustment off throttle (percentage)
			float     m_frontCamber;              // Front camber angle (suspension geometry)
			float     m_rearCamber;               // Rear camber angle (suspension geometry)
			float     m_frontToe;                 // Front toe angle (suspension geometry)
			float     m_rearToe;                  // Rear toe angle (suspension geometry)
			uint8     m_frontSuspension;          // Front suspension
			uint8     m_rearSuspension;           // Rear suspension
			uint8     m_frontAntiRollBar;         // Front anti-roll bar
			uint8     m_rearAntiRollBar;          // Front anti-roll bar
			uint8     m_frontSuspensionHeight;    // Front ride height
			uint8     m_rearSuspensionHeight;     // Rear ride height
			uint8     m_brakePressure;            // Brake pressure (percentage)
			uint8     m_brakeBias;                // Brake bias (percentage)
			float     m_frontTyrePressure;        // Front tyre pressure (PSI)
			float     m_rearTyrePressure;         // Rear tyre pressure (PSI)
			uint8     m_ballast;                  // Ballast
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
			uint16    m_speed;                    // Speed of car in kilometres per hour
			float     m_throttle;                 // Amount of throttle applied (0.0 to 1.0)
			float     m_steer;                    // Steering (-1.0 (full lock left) to 1.0 (full lock right))
			float     m_brake;                    // Amount of brake applied (0.0 to 1.0)
			uint8     m_clutch;                   // Amount of clutch applied (0 to 100)
			int8      m_gear;                     // Gear selected (1-8, N=0, R=-1)
			uint16    m_engineRPM;                // Engine RPM
			uint8     m_drs;                      // 0 = off, 1 = on
			uint8     m_revLightsPercent;         // Rev lights indicator (percentage)
			uint16    m_brakesTemperature[4];     // Brakes temperature (celsius)
			uint16    m_tyresSurfaceTemperature[4]; // Tyres surface temperature (celsius)
			uint16    m_tyresInnerTemperature[4]; // Tyres inner temperature (celsius)
			uint16    m_engineTemperature;        // Engine temperature (celsius)
			float     m_tyresPressure[4];         // Tyres pressure (PSI)
			uint8     m_surfaceType[4];           // Driving surface, see appendices
		}
	);
	PACK(
		struct PacketCarTelemetryData
		{
			PacketHeader    	m_header;	      // Header

			CarTelemetryData	m_carTelemetryData[20];

			uint32              m_buttonStatus;        // Bit flags specifying which buttons are being pressed
													// currently - see appendices
		}
	);
	PACK(
		struct CarStatusData
		{
			uint8       m_tractionControl;          // 0 (off) - 2 (high)
			uint8       m_antiLockBrakes;           // 0 (off) - 1 (on)
			uint8       m_fuelMix;                  // Fuel mix - 0 = lean, 1 = standard, 2 = rich, 3 = max
			uint8       m_frontBrakeBias;           // Front brake bias (percentage)
			uint8       m_pitLimiterStatus;         // Pit limiter status - 0 = off, 1 = on
			float       m_fuelInTank;               // Current fuel mass
			float       m_fuelCapacity;             // Fuel capacity
			float       m_fuelRemainingLaps;        // Fuel remaining in terms of laps (value on MFD)
			uint16      m_maxRPM;                   // Cars max RPM, point of rev limiter
			uint16      m_idleRPM;                  // Cars idle RPM
			uint8       m_maxGears;                 // Maximum number of gears
			uint8       m_drsAllowed;               // 0 = not allowed, 1 = allowed, -1 = unknown
			uint8       m_tyresWear[4];             // Tyre wear percentage
			uint8       m_actualTyreCompound;	   // F1 Modern - 16 = C5, 17 = C4, 18 = C3, 19 = C2, 20 = C1
																// 7 = inter, 8 = wet
																// F1 Classic - 9 = dry, 10 = wet
																// F2 – 11 = super soft, 12 = soft, 13 = medium, 14 = hard
																// 15 = wet
			uint8       m_tyreVisualCompound;       // F1 visual (can be different from actual compound)
													// 16 = soft, 17 = medium, 18 = hard, 7 = inter, 8 = wet
													// F1 Classic – same as above
													// F2 – same as above
			uint8       m_tyresDamage[4];           // Tyre damage (percentage)
			uint8       m_frontLeftWingDamage;      // Front left wing damage (percentage)
			uint8       m_frontRightWingDamage;     // Front right wing damage (percentage)
			uint8       m_rearWingDamage;           // Rear wing damage (percentage)
			uint8       m_engineDamage;             // Engine damage (percentage)
			uint8       m_gearBoxDamage;            // Gear box damage (percentage)
			int8        m_vehicleFiaFlags;	   // -1 = invalid/unknown, 0 = none, 1 = green
													// 2 = blue, 3 = yellow, 4 = red
			float       m_ersStoreEnergy;           // ERS energy store in Joules
			uint8       m_ersDeployMode;            // ERS deployment mode, 0 = none, 1 = low, 2 = medium
							// 3 = high, 4 = overtake, 5 = hotlap
			float       m_ersHarvestedThisLapMGUK;  // ERS energy harvested this lap by MGU-K
			float       m_ersHarvestedThisLapMGUH;  // ERS energy harvested this lap by MGU-H
			float       m_ersDeployedThisLap;       // ERS energy deployed this lap
		}
	);
	PACK(
		struct PacketCarStatusData
		{
			PacketHeader    	m_header;	   // Header

			CarStatusData	m_carStatusData[20];
		}
	);
}	
}
#endif
