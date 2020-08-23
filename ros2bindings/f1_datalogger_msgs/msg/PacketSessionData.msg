PacketHeader header                	 # Header
uint8 weather               	 # Weather - 0 = clear, 1 = light cloud, 2 = overcast
													 # 3 = light rain, 4 = heavy rain, 5 = storm

int8 track_temperature     	 # Track temp. in degrees celsius
int8 air_temperature       	 # Air temp. in degrees celsius
uint8 total_laps            	 # Total number of laps in this race
uint16 track_length            	 # Track length in metres
uint8 session_type          	 # 0 = unknown, 1 = P1, 2 = P2, 3 = P3, 4 = Short P
											 # 5 = Q1, 6 = Q2, 7 = Q3, 8 = Short Q, 9 = OSQ
											 # 10 = R, 11 = R2, 12 = Time Trial
int8 track_id # -1 for unknown, 0-21 for tracks, see appendix
uint8 era # Era, 0 = modern, 1 = classic
uint16 session_time_left     	 # Time left in session in seconds
uint16 session_duration      	 # Session duration in seconds
uint8 pit_speed_limit       	 # Pit speed limit in kilometres per hour
uint8 game_paused # Whether the game is paused
uint8 is_spectating         	 # Whether the player is spectating
uint8 spectator_car_index   	 # Index of the car being spectated
uint8 sli_pro_native_support 	 # SLI Pro support, 0 = inactive, 1 = active
uint8 num_marshal_zones          	 # Number of marshal zones to follow
MarshalZone[21] marshal_zones # List of marshal zones – max 21
uint8 safety_car_status # 0 = no safety car, 1 = full safety car
											 # 2 = virtual safety car
uint8 network_game # 0 = offline, 1 = online