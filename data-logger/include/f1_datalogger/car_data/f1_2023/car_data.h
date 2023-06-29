#ifndef INCLUDE_F1_2023_CAR_DATA_H
#define INCLUDE_F1_2023_CAR_DATA_H
#include <f1_datalogger/car_data/typedefs.h>

namespace deepf1{
namespace twenty_twentythree{

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
		LOBBY_INFO=9,
		MOTION_DATA_EX=13
	};

	PACK(
	struct PacketHeader
	{
		uint16 packetFormat;            // 2023
		uint8 gameYear;                // Game year - last two digits e.g. 23
		uint8 gameMajorVersion;        // Game major version - "X.00"
		uint8 gameMinorVersion;        // Game minor version - "1.XX"
		uint8 packetVersion;           // Version of this packet type, all start from 1
		uint8 packetId;                // Identifier for the packet type, see below
		uint64 sessionUID;              // Unique identifier for the session
		float sessionTime;             // Session timestamp
		uint32 frameIdentifier;         // Identifier for the frame the data was retrieved on
		uint32 overallFrameIdentifier;  // Overall identifier for the frame the data was retrieved on, doesn't go back after flashbacks
		uint8 playerCarIndex;          // Index of player's car in the array
		uint8 secondaryPlayerCarIndex; // Index of secondary player's car in the array (splitscreen) 255 if no second player
											
	}
	);
	PACK(
	struct CarMotionData
	{
		float worldPositionX;           // World space X position - metres
		float worldPositionY;           // World space Y position
		float worldPositionZ;           // World space Z position
		float worldVelocityX;           // Velocity in world space X – metres/s
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
	}
	);
	PACK(
	struct PacketMotionExData
	{
		PacketHeader header;               	// Header

		// Extra player car ONLY data
		float suspensionPosition[4];       // Note: All wheel arrays have the following order:
		float suspensionVelocity[4];       // RL, RR, FL, FR
		float suspensionAcceleration[4];	// RL, RR, FL, FR
		float wheelSpeed[4];           	// Speed of each wheel
		float wheelSlipRatio[4];           // Slip ratio for each wheel
		float wheelSlipAngle[4];           // Slip angles for each wheel
		float wheelLatForce[4];            // Lateral forces for each wheel
		float wheelLongForce[4];           // Longitudinal forces for each wheel
		float heightOfCOGAboveGround;      // Height of centre of gravity above ground    
		float localVelocityX;         	// Velocity in local space – metres/s
		float localVelocityY;         	// Velocity in local space
		float localVelocityZ;         	// Velocity in local space
		float angularVelocityX;		// Angular velocity x-component – radians/s
		float angularVelocityY;            // Angular velocity y-component
		float angularVelocityZ;            // Angular velocity z-component
		float angularAccelerationX;        // Angular acceleration x-component – radians/s/s
		float angularAccelerationY;	// Angular acceleration y-component
		float angularAccelerationZ;        // Angular acceleration z-component
		float frontWheelsAngle;            // Current front wheels angle in radians
		float wheelVertForce[4];           // Vertical forces for each wheel
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
		uint8 sessionType;              // 0 = unknown, 1 = P1, 2 = P2, 3 = P3, 4 = Short P, 5 = Q1
                                          // 6 = Q2, 7 = Q3, 8 = Short Q, 9 = OSQ, 10 = R, 11 = R2
                                          // 12 = R3, 13 = Time Trial
		uint8 timeOffset;               // Time in minutes the forecast is for
		uint8 weather;                  // Weather - 0 = clear, 1 = light cloud, 2 = overcast
											// 3 = light rain, 4 = heavy rain, 5 = storm
		int8 trackTemperature;         // Track temp. in degrees Celsius
		int8 trackTemperatureChange;   // Track temp. change – 0 = up, 1 = down, 2 = no change
		int8 airTemperature;           // Air temp. in degrees celsius
		int8 airTemperatureChange;     // Air temp. change – 0 = up, 1 = down, 2 = no change
		uint8 rainPercentage;           // Rain percentage (0-100)
	}
	);

	PACK(
	struct PacketSessionData
	{
		PacketHeader header;               	// Header

		uint8 weather;              	// Weather - 0 = clear, 1 = light cloud, 2 = overcast
													// 3 = light rain, 4 = heavy rain, 5 = storm
		int8 trackTemperature;    	// Track temp. in degrees celsius
		int8 airTemperature;      	// Air temp. in degrees celsius
		uint8 totalLaps;           	// Total number of laps in this race
		uint16 trackLength;           	// Track length in metres
		uint8 sessionType;         	// 0 = unknown, 1 = P1, 2 = P2, 3 = P3, 4 = Short P
													// 5 = Q1, 6 = Q2, 7 = Q3, 8 = Short Q, 9 = OSQ
													// 10 = R, 11 = R2, 12 = R3, 13 = Time Trial
		int8  trackId;         		// -1 for unknown, see appendix
		uint8 formula;                  	// Formula, 0 = F1 Modern, 1 = F1 Classic, 2 = F2,
													// 3 = F1 Generic, 4 = Beta, 5 = Supercars
	// 6 = Esports, 7 = F2 2021
		uint16 sessionTimeLeft;    	// Time left in session in seconds
		uint16 sessionDuration;     	// Session duration in seconds
		uint8 pitSpeedLimit;      	// Pit speed limit in kilometres per hour
		uint8 gamePaused;                // Whether the game is paused – network game only
		uint8 isSpectating;        	// Whether the player is spectating
		uint8 spectatorCarIndex;  	// Index of the car being spectated
		uint8 sliProNativeSupport;	// SLI Pro support, 0 = inactive, 1 = active
		uint8 numMarshalZones;         	// Number of marshal zones to follow
		MarshalZone marshalZones[21];         	// List of marshal zones – max 21
		uint8 safetyCarStatus;           // 0 = no safety car, 1 = full
													// 2 = virtual, 3 = formation lap
		uint8 networkGame;               // 0 = offline, 1 = online
		uint8 numWeatherForecastSamples; // Number of weather samples to follow
		WeatherForecastSample weatherForecastSamples[56];   // Array of weather forecast samples
		uint8 forecastAccuracy;          // 0 = Perfect, 1 = Approximate
		uint8 aiDifficulty;              // AI Difficulty rating – 0-110
		uint32 seasonLinkIdentifier;      // Identifier for season - persists across saves
		uint32 weekendLinkIdentifier;     // Identifier for weekend - persists across saves
		uint32 sessionLinkIdentifier;     // Identifier for session - persists across saves
		uint8 pitStopWindowIdealLap;     // Ideal lap to pit on for current strategy (player)
		uint8 pitStopWindowLatestLap;    // Latest lap to pit on for current strategy (player)
		uint8 pitStopRejoinPosition;     // Predicted position to rejoin at (player)
		uint8 steeringAssist;            // 0 = off, 1 = on
		uint8 brakingAssist;             // 0 = off, 1 = low, 2 = medium, 3 = high
		uint8 gearboxAssist;             // 1 = manual, 2 = manual & suggested gear, 3 = auto
		uint8 pitAssist;                 // 0 = off, 1 = on
		uint8 pitReleaseAssist;          // 0 = off, 1 = on
		uint8 ERSAssist;                 // 0 = off, 1 = on
		uint8 DRSAssist;                 // 0 = off, 1 = on
		uint8 dynamicRacingLine;         // 0 = off, 1 = corners only, 2 = full
		uint8 dynamicRacingLineType;     // 0 = 2D, 1 = 3D
		uint8 gameMode;                  // Game mode id - see appendix
		uint8 ruleSet;                   // Ruleset - see appendix
		uint32 timeOfDay;                 // Local time of day - minutes since midnight
		uint8 sessionLength;             // 0 = None, 2 = Very Short, 3 = Short, 4 = Medium
	// 5 = Medium Long, 6 = Long, 7 = Full
		uint8 speedUnitsLeadPlayer;             // 0 = MPH, 1 = KPH
		uint8 temperatureUnitsLeadPlayer;       // 0 = Celsius, 1 = Fahrenheit
		uint8 speedUnitsSecondaryPlayer;        // 0 = MPH, 1 = KPH
		uint8 temperatureUnitsSecondaryPlayer;  // 0 = Celsius, 1 = Fahrenheit
		uint8 numSafetyCarPeriods;              // Number of safety cars called during session
		uint8 numVirtualSafetyCarPeriods;       // Number of virtual safety cars called
		uint8 numRedFlagPeriods;                // Number of red flags called during session

	}
	);

	PACK(
	struct LapData
	{
		uint32 lastLapTimeInMS;	       	 // Last lap time in milliseconds
		uint32 currentLapTimeInMS; 	 // Current time around the lap in milliseconds
		uint16 sector1TimeInMS;           // Sector 1 time in milliseconds
		uint8 sector1TimeMinutes;        // Sector 1 whole minute part
		uint16 sector2TimeInMS;           // Sector 2 time in milliseconds
		uint8 sector2TimeMinutes;        // Sector 2 whole minute part
		uint16 deltaToCarInFrontInMS;     // Time delta to car in front in milliseconds
		uint16 deltaToRaceLeaderInMS;     // Time delta to race leader in milliseconds
		float lapDistance;		 // Distance vehicle is around current lap in metres – could
						// be negative if line hasn’t been crossed yet
		float totalDistance;		 // Total distance travelled in session in metres – could
						// be negative if line hasn’t been crossed yet
		float safetyCarDelta;            // Delta in seconds for safety car
		uint8 carPosition;   	         // Car race position
		uint8 currentLapNum;		 // Current lap number
		uint8 pitStatus;            	 // 0 = none, 1 = pitting, 2 = in pit area
		uint8 numPitStops;            	 // Number of pit stops taken in this race
		uint8 sector;               	 // 0 = sector1, 1 = sector2, 2 = sector3
		uint8 currentLapInvalid;    	 // Current lap invalid - 0 = valid, 1 = invalid
		uint8 penalties;            	 // Accumulated time penalties in seconds to be added
		uint8 totalWarnings;             // Accumulated number of warnings issued
		uint8 cornerCuttingWarnings;     // Accumulated number of corner cutting warnings issued
		uint8 numUnservedDriveThroughPens;  // Num drive through pens left to serve
		uint8 numUnservedStopGoPens;        // Num stop go pens left to serve
		uint8 gridPosition;         	 // Grid position the vehicle started the race in
		uint8 driverStatus;         	 // Status of driver - 0 = in garage, 1 = flying lap
											// 2 = in lap, 3 = out lap, 4 = on track
		uint8 resultStatus;              // Result status - 0 = invalid, 1 = inactive, 2 = active
											// 3 = finished, 4 = didnotfinish, 5 = disqualified
											// 6 = not classified, 7 = retired
		uint8 pitLaneTimerActive;     	 // Pit lane timing, 0 = inactive, 1 = active
		uint16 pitLaneTimeInLaneInMS;   	 // If active, the current time spent in the pit lane in ms
		uint16 pitStopTimerInMS;        	 // Time of the actual pit stop in ms
		uint8 pitStopShouldServePen;   	 // Whether the car should serve a penalty at this stop

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
			uint8 penaltyType;		// Penalty type – see Appendices
			uint8 infringementType;		// Infringement type – see Appendices
			uint8 vehicleIdx;         	// Vehicle index of the car the penalty is applied to
			uint8 otherVehicleIdx;    	// Vehicle index of the other car involved
			uint8 time;               	// Time gained, or time spent doing action in seconds
			uint8 lapNum;             	// Lap the penalty occurred on
			uint8 placesGained;       	// Number of places gained by this
		} Penalty;

		struct
		{
			uint8 vehicleIdx;		// Vehicle index of the vehicle triggering speed trap
			float speed;      		// Top speed achieved in kilometres per hour
			uint8 isOverallFastestInSession; // Overall fastest speed in session = 1, otherwise 0
			uint8 isDriverFastestInSession;  // Fastest speed for driver in session = 1, otherwise 0
			uint8 fastestVehicleIdxInSession;// Vehicle index of the vehicle that is the fastest
	// in this session
			float fastestSpeedInSession;      // Speed of the vehicle that is the fastest
	// in this session
		} SpeedTrap;

		struct
		{
			uint8 numLights;			// Number of lights showing
		} StartLIghts;

		struct
		{
			uint8 vehicleIdx;                 // Vehicle index of the vehicle serving drive through
		} DriveThroughPenaltyServed;

		struct
		{
			uint8 vehicleIdx;                 // Vehicle index of the vehicle serving stop go
		} StopGoPenaltyServed;

		struct
		{
			uint32 flashbackFrameIdentifier;  // Frame identifier flashed back to
			float flashbackSessionTime;       // Session time flashed back to
		} Flashback;

		struct
		{
			uint32 buttonStatus;              // Bit flags specifying which buttons are being pressed
											// currently - see appendices
		} Buttons;

		struct
		{
			uint8 overtakingVehicleIdx;       // Vehicle index of the vehicle overtaking
			uint8 beingOvertakenVehicleIdx;   // Vehicle index of the vehicle being overtaken
		} Overtake;

		}
	);
	PACK(
	struct PacketEventData
	{
		PacketHeader    	 header;               	// Header
    
		uint8           	 eventStringCode[4];   	// Event string code, see below
		EventDataDetails	 eventDetails;         	// Event details - should be interpreted differently
													// for each type
	}
	);
	PACK(
	struct ParticipantData
	{
		uint8 aiControlled;      // Whether the vehicle is AI (1) or Human (0) controlled
		uint8 driverId;	   // Driver id - see appendix, 255 if network human
		uint8 networkId;	   // Network id – unique identifier for network players
		uint8 teamId;            // Team id - see appendix
		uint8 myTeam;            // My team flag – 1 = My Team, 0 = otherwise
		uint8 raceNumber;        // Race number of the car
		uint8 nationality;       // Nationality of the driver
		char name[48];          // Name of participant in UTF-8 format – null terminated
					// Will be truncated with … (U+2026) if too long
		uint8 yourTelemetry;     // The player's UDP setting, 0 = restricted, 1 = public
		uint8 showOnlineNames;   // The player's show online names setting, 0 = off, 1 = on
		uint8 platform;          // 1 = Steam, 3 = PlayStation, 4 = Xbox, 6 = Origin, 255 = unknown

	}
	);
	PACK(
	struct PacketParticipantsData
	{
		PacketHeader header;            // Header

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
		uint16 speed;                    // Speed of car in kilometres per hour
		float throttle;                 // Amount of throttle applied (0.0 to 1.0)
		float steer;                    // Steering (-1.0 (full lock left) to 1.0 (full lock right))
		float brake;                    // Amount of brake applied (0.0 to 1.0)
		uint8 clutch;                   // Amount of clutch applied (0 to 100)
		int8 gear;                     // Gear selected (1-8, N=0, R=-1)
		uint16 engineRPM;                // Engine RPM
		uint8 drs;                      // 0 = off, 1 = on
		uint8 revLightsPercent;         // Rev lights indicator (percentage)
		uint16 revLightsBitValue;        // Rev lights (bit 0 = leftmost LED, bit 14 = rightmost LED)
		uint16 brakesTemperature[4];     // Brakes temperature (celsius)
		uint8 tyresSurfaceTemperature[4]; // Tyres surface temperature (celsius)
		uint8 tyresInnerTemperature[4]; // Tyres inner temperature (celsius)
		uint16 engineTemperature;        // Engine temperature (celsius)
		float tyresPressure[4];         // Tyres pressure (PSI)
		uint8 surfaceType[4];           // Driving surface, see appendices
	}
	);
	PACK(
	struct PacketCarTelemetryData
	{
		PacketHeader header;	      // Header
		CarTelemetryData carTelemetryData[22];
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
		uint8 tractionControl;          // Traction control - 0 = off, 1 = medium, 2 = full
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
		uint8 drsAllowed;               // 0 = not allowed, 1 = allowed
		uint16 drsActivationDistance;    // 0 = DRS not available, non-zero - DRS will be available
												// in [X] metres
		uint8 actualTyreCompound;	   // F1 Modern - 16 = C5, 17 = C4, 18 = C3, 19 = C2, 20 = C1
						// 21 = C0, 7 = inter, 8 = wet
						// F1 Classic - 9 = dry, 10 = wet
						// F2 – 11 = super soft, 12 = soft, 13 = medium, 14 = hard
						// 15 = wet
		uint8 visualTyreCompound;       // F1 visual (can be different from actual compound)
												// 16 = soft, 17 = medium, 18 = hard, 7 = inter, 8 = wet
												// F1 Classic – same as above
												// F2 ‘19, 15 = wet, 19 – super soft, 20 = soft
												// 21 = medium , 22 = hard
		uint8 tyresAgeLaps;             // Age in laps of the current set of tyres
		int8 vehicleFiaFlags;	   // -1 = invalid/unknown, 0 = none, 1 = green
												// 2 = blue, 3 = yellow
		float enginePowerICE;           // Engine power output of ICE (W)
		float enginePowerMGUK;          // Engine power output of MGU-K (W)
		float ersStoreEnergy;           // ERS energy store in Joules
		uint8 ersDeployMode;            // ERS deployment mode, 0 = none, 1 = medium
						// 2 = hotlap, 3 = overtake
		float ersHarvestedThisLapMGUK;  // ERS energy harvested this lap by MGU-K
		float ersHarvestedThisLapMGUH;  // ERS energy harvested this lap by MGU-H
		float ersDeployedThisLap;       // ERS energy deployed this lap
		uint8 networkPaused;            // Whether the car is paused in a network game
	}
	);
	PACK(
	struct PacketCarStatusData
	{
		PacketHeader header;	   // Header
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
										// 3 = finished, 4 = didnotfinish, 5 = disqualified
										// 6 = not classified, 7 = retired
		uint32 bestLapTimeInMS;       // Best lap time of the session in milliseconds
		double totalRaceTime;         // Total race time in seconds without penalties
		uint8 penaltiesTime;         // Total penalties accumulated in seconds
		uint8 numPenalties;          // Number of penalties applied to this driver
		uint8 numTyreStints;         // Number of tyres stints up to maximum
		uint8 tyreStintsActual[8];   // Actual tyres used by this driver
		uint8 tyreStintsVisual[8];   // Visual tyres used by this driver
		uint8 tyreStintsEndLaps[8];  // The lap number stints end on
	}
	);
	PACK(
	struct PacketFinalClassificationData
	{
		PacketHeader header;                      // Header

		uint8 numCars;          // Number of cars in the final classification
		FinalClassificationData classificationData[22];
	}
	);


	PACK(
	struct LobbyInfoData
	{
		uint8 aiControlled;      // Whether the vehicle is AI (1) or Human (0) controlled
		uint8 teamId;            // Team id - see appendix (255 if no team currently selected)
		uint8 nationality;       // Nationality of the driver
		uint8 platform;          // 1 = Steam, 3 = PlayStation, 4 = Xbox, 6 = Origin, 255 = unknown
		char name[48];	  // Name of participant in UTF-8 format – null terminated
									// Will be truncated with ... (U+2026) if too long
		uint8 carNumber;         // Car number of the player
		uint8 readyStatus;       // 0 = not ready, 1 = ready, 2 = spectating
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
