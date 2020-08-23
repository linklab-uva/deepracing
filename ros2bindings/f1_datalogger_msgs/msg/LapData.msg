float32 last_lap_time           # Last lap time in seconds
float32 current_lap_time        # Current time around the lap in seconds
float32 best_lap_time           # Best lap time of the session in seconds
float32 sector1_time           # Sector 1 time in seconds
float32 sector2_time          # Sector 2 time in seconds
float32 lap_distance          # Distance vehicle is around current lap in metres – could
                                        # be negative if line hasn’t been crossed yet
float32 total_distance         # Total distance travelled in session in metres – could
                                        # be negative if line hasn’t been crossed yet
float32 safety_car_delta        # Delta in seconds for safety car
uint8 car_position           # Car race position
uint8 current_lap_num         # Current lap number
uint8 pit_status             # 0 = none, 1 = pitting, 2 = in pit area
uint8 sector                # 0 = sector1, 1 = sector2, 2 = sector3
uint8 current_lap_invalid     # Current lap invalid - 0 = valid, 1 = invalid
uint8 penalties             # Accumulated time penalties in seconds to be added
uint8 grid_position          # Grid position the vehicle started the race in
uint8 driver_status          # Status of driver - 0 = in garage, 1 = flying lap
                                        # 2 = in lap, 3 = out lap, 4 = on track
uint8 result_status          # Result status - 0 = invalid, 1 = inactive, 2 = active
                                        # 3 = finished, 4 = disqualified, 5 = not classified
											 # 6 = retired