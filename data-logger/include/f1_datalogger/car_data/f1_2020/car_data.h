#ifndef INCLUDE_F1_2020_CAR_DATA_H
#define INCLUDE_F1_2020_CAR_DATA_H
#include <f1_datalogger/car_data/typedefs.h>

namespace deepf1{
namespace twenty_twenty{

	enum PacketID
	{
		MOTION=0,
		SESSION=1,
		LAPDATA=2,
		EVENT=3,
		PARTICIPANTS=4,
		CARSETUPS=5,
		CARTELEMETRY=6,
		CARSTATUS=7,
		FINAL_CLASSIFICATION=8,
		LOBBY_INFO=9
	};

	PACK(
	struct PacketHeader
	{
		uint16 packetFormat;             // 2020
		uint8 gameMajorVersion;         // Game major version - "X.00"
		uint8 gameMinorVersion;         // Game minor version - "1.XX"
		uint8 packetVersion;            // Version of this packet type, all start from 1
		uint8 packetId;                 // Identifier for the packet type, see below
		uint64 sessionUID;               // Unique identifier for the session
		float sessionTime;              // Session timestamp
		uint32 frameIdentifier;          // Identifier for the frame the data was retrieved on
		uint8 playerCarIndex;           // Index of player's car in the array
		
	// ADDED IN BETA 2: 
		uint8 secondaryPlayerCarIndex;  // Index of secondary player's car in the array (splitscreen)
											// 255 if no second player
	}
	);
	PACK(
	struct CarMotionData
	{
		float worldPositionX;           // World space X position
		float worldPositionY;           // World space Y position
		float worldPositionZ;           // World space Z position
		float worldVelocityX;           // Velocity in world space X
		float worldVelocityY;           // Velocity in world space Y
		float worldVelocityZ;           // Velocity in world space Z
		int16 worldForwardDirX;         // World space forward X direction (normalised)
		int16 worldForwardDirY;         // World space forward Y direction (normalised)
		int16 worldForwardDirZ;         // World space forward Z direction (normalised)
		int16 worldRightDirX;           // World space right X direction (normalised)
		int16 worldRightDirY;           // World space right Y direction (normalised)
		int16 worldRightDirZ;           // World space right Z direction (normalised)
		float gForceLateral;            // Lateral G-Force component
		float gForceLongitudinal;       // Longitudinal G-Force component
		float gForceVertical;           // Vertical G-Force component
		float yaw;                      // Yaw angle in radians
		float pitch;                    // Pitch angle in radians
		float roll;                     // Roll angle in radians
	}
	);
	PACK(
	struct PacketMotionData
	{
		PacketHeader header;               	// Header

		CarMotionData carMotionData[22];    	// Data for all cars on track

		// Extra player car ONLY data
		float suspensionPosition[4];      // Note: All wheel arrays have the following order:
		float suspensionVelocity[4];      // RL, RR, FL, FR
		float suspensionAcceleration[4];	// RL, RR, FL, FR
		float wheelSpeed[4];           	// Speed of each wheel
		float wheelSlip[4];               // Slip ratio for each wheel
		float localVelocityX;         	// Velocity in local space
		float localVelocityY;         	// Velocity in local space
		float localVelocityZ;         	// Velocity in local space
		float angularVelocityX;		    // Angular velocity x-component
		float angularVelocityY;           // Angular velocity y-component
		float angularVelocityZ;           // Angular velocity z-component
		float angularAccelerationX;       // Angular velocity x-component
		float angularAccelerationY;	    // Angular velocity y-component
		float angularAccelerationZ;       // Angular velocity z-component
		float frontWheelsAngle;           // Current front wheels angle in radians
	}
	);

	PACK(
	struct MarshalZone
	{
		float zoneStart;   // Fraction (0..1) of way through the lap the marshal zone starts
		int8 zoneFlag;    // -1 = invalid/unknown, 0 = none, 1 = green, 2 = blue, 3 = yellow, 4 = red
	}
	);
	
	PACK(
	struct WeatherForecastSample
	{
		uint8 sessionType;                     // 0 = unknown, 1 = P1, 2 = P2, 3 = P3, 4 = Short P, 5 = Q1
													// 6 = Q2, 7 = Q3, 8 = Short Q, 9 = OSQ, 10 = R, 11 = R2
													// 12 = Time Trial
		uint8 timeOffset;                      // Time in minutes the forecast is for
		uint8 weather;                         // Weather - 0 = clear, 1 = light cloud, 2 = overcast
													// 3 = light rain, 4 = heavy rain, 5 = storm
		int8 trackTemperature;                // Track temp. in degrees celsius
		int8 airTemperature;                  // Air temp. in degrees celsius
	}
	);

	PACK(
	struct PacketSessionData
	{
		PacketHeader header;                    // Header

		uint8 weather;                   // Weather - 0 = clear, 1 = light cloud, 2 = overcast
													// 3 = light rain, 4 = heavy rain, 5 = storm
		int8 trackTemperature;          // Track temp. in degrees celsius
		int8 airTemperature;            // Air temp. in degrees celsius
		uint8 totalLaps;                 // Total number of laps in this race
		uint16 trackLength;               // Track length in metres
		uint8 sessionType;               // 0 = unknown, 1 = P1, 2 = P2, 3 = P3, 4 = Short P
													// 5 = Q1, 6 = Q2, 7 = Q3, 8 = Short Q, 9 = OSQ
													// 10 = R, 11 = R2, 12 = Time Trial
		int8 trackId;                   // -1 for unknown, 0-21 for tracks, see appendix
		uint8 formula;                   // Formula, 0 = F1 Modern, 1 = F1 Classic, 2 = F2,
													// 3 = F1 Generic
		uint16 sessionTimeLeft;           // Time left in session in seconds
		uint16 sessionDuration;           // Session duration in seconds
		uint8 pitSpeedLimit;             // Pit speed limit in kilometres per hour
		uint8 gamePaused;                // Whether the game is paused
		uint8 isSpectating;              // Whether the player is spectating
		uint8 spectatorCarIndex;         // Index of the car being spectated
		uint8 sliProNativeSupport;	 // SLI Pro support, 0 = inactive, 1 = active
		uint8 numMarshalZones;           // Number of marshal zones to follow
		MarshalZone marshalZones[21];          // List of marshal zones – max 21
		uint8 safetyCarStatus;           // 0 = no safety car, 1 = full safety car
													// 2 = virtual safety car
		uint8 networkGame;               // 0 = offline, 1 = online
		uint8 numWeatherForecastSamples; // Number of weather samples to follow
		WeatherForecastSample weatherForecastSamples[20];   // Array of weather forecast samples
	}
	);

	PACK(
	struct LapData
	{
		float lastLapTime;               // Last lap time in seconds
		float currentLapTime;            // Current time around the lap in seconds
	
		//UPDATED in Beta 3:
		uint16 sector1TimeInMS;           // Sector 1 time in milliseconds
		uint16 sector2TimeInMS;           // Sector 2 time in milliseconds
		float bestLapTime;               // Best lap time of the session in seconds
		uint8 bestLapNum;                // Lap number best time achieved on
		uint16 bestLapSector1TimeInMS;    // Sector 1 time of best lap in the session in milliseconds
		uint16 bestLapSector2TimeInMS;    // Sector 2 time of best lap in the session in milliseconds
		uint16 bestLapSector3TimeInMS;    // Sector 3 time of best lap in the session in milliseconds
		uint16 bestOverallSector1TimeInMS;// Best overall sector 1 time of the session in milliseconds
		uint8 bestOverallSector1LapNum;  // Lap number best overall sector 1 time achieved on
		uint16 bestOverallSector2TimeInMS;// Best overall sector 2 time of the session in milliseconds
		uint8 bestOverallSector2LapNum;  // Lap number best overall sector 2 time achieved on
		uint16 bestOverallSector3TimeInMS;// Best overall sector 3 time of the session in milliseconds
		uint8 bestOverallSector3LapNum;  // Lap number best overall sector 3 time achieved on
	
	
		float lapDistance;               // Distance vehicle is around current lap in metres – could
											// be negative if line hasn’t been crossed yet
		float totalDistance;             // Total distance travelled in session in metres – could
											// be negative if line hasn’t been crossed yet
		float safetyCarDelta;            // Delta in seconds for safety car
		uint8 carPosition;               // Car race position
		uint8 currentLapNum;             // Current lap number
		uint8 pitStatus;                 // 0 = none, 1 = pitting, 2 = in pit area
		uint8 sector;                    // 0 = sector1, 1 = sector2, 2 = sector3
		uint8 currentLapInvalid;         // Current lap invalid - 0 = valid, 1 = invalid
		uint8 penalties;                 // Accumulated time penalties in seconds to be added
		uint8 gridPosition;              // Grid position the vehicle started the race in
		uint8 driverStatus;              // Status of driver - 0 = in garage, 1 = flying lap
											// 2 = in lap, 3 = out lap, 4 = on track
		uint8 resultStatus;              // Result status - 0 = invalid, 1 = inactive, 2 = active
											// 3 = finished, 4 = disqualified, 5 = not classified
											// 6 = retired
	}
	);
	PACK(
	struct PacketLapData
	{
		PacketHeader header;             // Header

		LapData lapData[22];        // Lap data for all cars on track
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

		struct
		{
			uint8 penaltyType;          // Penalty type – see Appendices
			uint8 infringementType;     // Infringement type – see Appendices
			uint8 vehicleIdx;           // Vehicle index of the car the penalty is applied to
			uint8 otherVehicleIdx;      // Vehicle index of the other car involved
			uint8 time;                 // Time gained, or time spent doing action in seconds
			uint8 lapNum;               // Lap the penalty occurred on
			uint8 placesGained;         // Number of places gained by this
		} Penalty;

		struct
		{
			uint8 vehicleIdx; // Vehicle index of the vehicle triggering speed trap
			float speed;      // Top speed achieved in kilometres per hour
		} SpeedTrap;
	}
	);
	PACK(
	struct PacketEventData
	{
		PacketHeader header;             // Header
			
		uint8 eventStringCode[4]; // Event string code, see below
		EventDataDetails eventDetails;       // Event details - should be interpreted differently
												// for each type
	}
	);
	PACK(
	struct ParticipantData
	{
		uint8 aiControlled;           // Whether the vehicle is AI (1) or Human (0) controlled
		uint8 driverId;               // Driver id - see appendix
		uint8 teamId;                 // Team id - see appendix
		uint8 raceNumber;             // Race number of the car
		uint8 nationality;            // Nationality of the driver
		char name[48];               // Name of participant in UTF-8 format – null terminated
											// Will be truncated with … (U+2026) if too long
		uint8 yourTelemetry;          // The player's UDP setting, 0 = restricted, 1 = public
	}
	);
	PACK(
	struct PacketParticipantsData
	{
		PacketHeader header;           // Header

		uint8 numActiveCars;	// Number of active cars in the data – should match number of
											// cars on HUD
		ParticipantData participants[22];
	}
	);

	PACK(
	struct CarSetupData
	{
		uint8 frontWing;                // Front wing aero
		uint8 rearWing;                 // Rear wing aero
		uint8 onThrottle;               // Differential adjustment on throttle (percentage)
		uint8 offThrottle;              // Differential adjustment off throttle (percentage)
		float frontCamber;              // Front camber angle (suspension geometry)
		float rearCamber;               // Rear camber angle (suspension geometry)
		float frontToe;                 // Front toe angle (suspension geometry)
		float rearToe;                  // Rear toe angle (suspension geometry)
		uint8 frontSuspension;          // Front suspension
		uint8 rearSuspension;           // Rear suspension
		uint8 frontAntiRollBar;         // Front anti-roll bar
		uint8 rearAntiRollBar;          // Front anti-roll bar
		uint8 frontSuspensionHeight;    // Front ride height
		uint8 rearSuspensionHeight;     // Rear ride height
		uint8 brakePressure;            // Brake pressure (percentage)
		uint8 brakeBias;                // Brake bias (percentage)
		float rearLeftTyrePressure;     // Rear left tyre pressure (PSI)
		float rearRightTyrePressure;    // Rear right tyre pressure (PSI)
		float frontLeftTyrePressure;    // Front left tyre pressure (PSI)
		float frontRightTyrePressure;   // Front right tyre pressure (PSI)
		uint8 ballast;                  // Ballast
		float fuelLoad;                 // Fuel load
	}
	);
	PACK(
	struct PacketCarSetupData
	{
		PacketHeader header;            // Header

		CarSetupData carSetups[22];
	}
	);
	PACK(
	struct CarTelemetryData
	{
		uint16 speed;                         // Speed of car in kilometres per hour
		float throttle;                      // Amount of throttle applied (0.0 to 1.0)
		float steer;                         // Steering (-1.0 (full lock left) to 1.0 (full lock right))
		float brake;                         // Amount of brake applied (0.0 to 1.0)
		uint8 clutch;                        // Amount of clutch applied (0 to 100)
		int8 gear;                          // Gear selected (1-8, N=0, R=-1)
		uint16 engineRPM;                     // Engine RPM
		uint8 drs;                           // 0 = off, 1 = on
		uint8 revLightsPercent;              // Rev lights indicator (percentage)
		uint16 brakesTemperature[4];          // Brakes temperature (celsius)
		uint8 tyresSurfaceTemperature[4];    // Tyres surface temperature (celsius)
		uint8 tyresInnerTemperature[4];      // Tyres inner temperature (celsius)
		uint16 engineTemperature;             // Engine temperature (celsius)
		float tyresPressure[4];              // Tyres pressure (PSI)
		uint8 surfaceType[4];                // Driving surface, see appendices
	}
	);
	PACK(
	struct PacketCarTelemetryData
	{
		PacketHeader header;	       // Header

		CarTelemetryData carTelemetryData[22];

		uint32 buttonStatus;        // Bit flags specifying which buttons are being pressed
												// currently - see appendices

		// Added in Beta 3:
		uint8 mfdPanelIndex;       // Index of MFD panel open - 255 = MFD closed
												// Single player, race – 0 = Car setup, 1 = Pits
												// 2 = Damage, 3 =  Engine, 4 = Temperatures
												// May vary depending on game mode
		uint8 mfdPanelIndexSecondaryPlayer;   // See above
		int8 suggestedGear;       // Suggested gear for the player (1-8)
												// 0 if no gear suggested
	}
	);
	PACK(
	struct CarStatusData
	{
		uint8 tractionControl;          // 0 (off) - 2 (high)
		uint8 antiLockBrakes;           // 0 (off) - 1 (on)
		uint8 fuelMix;                  // Fuel mix - 0 = lean, 1 = standard, 2 = rich, 3 = max
		uint8 frontBrakeBias;           // Front brake bias (percentage)
		uint8 pitLimiterStatus;         // Pit limiter status - 0 = off, 1 = on
		float fuelInTank;               // Current fuel mass
		float fuelCapacity;             // Fuel capacity
		float fuelRemainingLaps;        // Fuel remaining in terms of laps (value on MFD)
		uint16 maxRPM;                   // Cars max RPM, point of rev limiter
		uint16 idleRPM;                  // Cars idle RPM
		uint8 maxGears;                 // Maximum number of gears
		uint8 drsAllowed;               // 0 = not allowed, 1 = allowed, -1 = unknown
		

		// Added in Beta3:
		uint16 drsActivationDistance;    // 0 = DRS not available, non-zero - DRS will be available
												// in [X] metres
		
		uint8 tyresWear[4];             // Tyre wear percentage
		uint8 actualTyreCompound;	    // F1 Modern - 16 = C5, 17 = C4, 18 = C3, 19 = C2, 20 = C1
							// 7 = inter, 8 = wet
							// F1 Classic - 9 = dry, 10 = wet
							// F2 – 11 = super soft, 12 = soft, 13 = medium, 14 = hard
							// 15 = wet
		uint8 visualTyreCompound;        // F1 visual (can be different from actual compound)
												// 16 = soft, 17 = medium, 18 = hard, 7 = inter, 8 = wet
												// F1 Classic – same as above
												// F2 – same as above
		uint8 tyresAgeLaps;             // Age in laps of the current set of tyres
		uint8 tyresDamage[4];           // Tyre damage (percentage)
		uint8 frontLeftWingDamage;      // Front left wing damage (percentage)
		uint8 frontRightWingDamage;     // Front right wing damage (percentage)
		uint8 rearWingDamage;           // Rear wing damage (percentage)
		
		// Added Beta 3:
		uint8 drsFault;                 // Indicator for DRS fault, 0 = OK, 1 = fault
		
		uint8 engineDamage;             // Engine damage (percentage)
		uint8 gearBoxDamage;            // Gear box damage (percentage)
		int8 vehicleFiaFlags;          // -1 = invalid/unknown, 0 = none, 1 = green
												// 2 = blue, 3 = yellow, 4 = red
		float ersStoreEnergy;           // ERS energy store in Joules
		uint8 ersDeployMode;            // ERS deployment mode, 0 = none, 1 = medium
												// 2 = overtake, 3 = hotlap
		float ersHarvestedThisLapMGUK;  // ERS energy harvested this lap by MGU-K
		float ersHarvestedThisLapMGUH;  // ERS energy harvested this lap by MGU-H
		float ersDeployedThisLap;       // ERS energy deployed this lap
	}
	);
	PACK(
	struct PacketCarStatusData
	{
		PacketHeader header;           // Header

		CarStatusData carStatusData[22];
	}
	);

	PACK(
	struct FinalClassificationData
	{
		uint8 position;              // Finishing position
		uint8 numLaps;               // Number of laps completed
		uint8 gridPosition;          // Grid position of the car
		uint8 points;                // Number of points scored
		uint8 numPitStops;           // Number of pit stops made
		uint8 resultStatus;          // Result status - 0 = invalid, 1 = inactive, 2 = active
										// 3 = finished, 4 = disqualified, 5 = not classified
										// 6 = retired
		float bestLapTime;           // Best lap time of the session in seconds
		double totalRaceTime;         // Total race time in seconds without penalties
		uint8 penaltiesTime;         // Total penalties accumulated in seconds
		uint8 numPenalties;          // Number of penalties applied to this driver
		uint8 numTyreStints;         // Number of tyres stints up to maximum
		uint8 tyreStintsActual[8];   // Actual tyres used by this driver
		uint8 tyreStintsVisual[8];   // Visual tyres used by this driver
	}
	);
	PACK(
	struct PacketFinalClassificationData
	{
		PacketHeader header;                             // Header

		uint8 numCars;                 // Number of cars in the final classification
		FinalClassificationData classificationData[22];
	}
	);


	PACK(
	struct LobbyInfoData
	{
		uint8 aiControlled;            // Whether the vehicle is AI (1) or Human (0) controlled
		uint8 teamId;                  // Team id - see appendix (255 if no team currently selected)
		uint8 nationality;             // Nationality of the driver
		char name[48];                // Name of participant in UTF-8 format – null terminated
											// Will be truncated with ... (U+2026) if too long
		uint8 readyStatus;             // 0 = not ready, 1 = ready, 2 = spectating
	}
	);
	PACK(
	struct PacketLobbyInfoData
	{
		PacketHeader header;                       // Header

		// Packet specific data
		uint8 numPlayers;               // Number of players in the lobby data
		LobbyInfoData lobbyPlayers[22];
	}
	);

	
}
}
#endif
