uint8 traction_control          # 0 (off) - 2 (high)
uint8 anti_lock_brakes           # 0 (off) - 1 (on)
uint8 fuel_mix                  # Fuel mix - 0 = lean, 1 = standard, 2 = rich, 3 = max
uint8 front_brake_bias           # Front brake bias (percentage)
uint8 pit_limiter_status         # Pit limiter status - 0 = off, 1 = on
float32 fuel_in_tank               # Current fuel mass
float32 fuel_capacity             # Fuel capacity
uint16 max_rpm                   # Cars max RPM, point of rev limiter
uint16 idle_rpm                  # Cars idle RPM
uint8 max_gears                 # Maximum number of gears
uint8 drs_allowed               # 0 = not allowed, 1 = allowed, -1 = unknown
uint8[4] tyres_wear             # Tyre wear percentage
uint8 tyre_compound             # Modern - 0 = hyper soft, 1 = ultra soft
                                        # 2 = super soft, 3 = soft, 4 = medium, 5 = hard
                                        # 6 = super hard, 7 = inter, 8 = wet
                                        # Classic - 0-6 = dry, 7-8 = wet
uint8[4] tyres_damage           # Tyre damage (percentage)
uint8 front_left_wing_damage      # Front left wing damage (percentage)
uint8 front_right_wing_damage     # Front right wing damage (percentage)
uint8 rear_wing_damage           # Rear wing damage (percentage)
uint8 engine_damage             # Engine damage (percentage)
uint8 gear_box_damage            # Gear box damage (percentage)
uint8 exhaust_damage            # Exhaust damage (percentage)
int8 vehicle_fia_flags          # -1 = invalid/unknown, 0 = none, 1 = green
                                        # 2 = blue, 3 = yellow, 4 = red
float32 ers_store_energy           # ERS energy store in Joules
uint8 ers_deploy_mode            # ERS deployment mode, 0 = none, 1 = low, 2 = medium
                                        # 3 = high, 4 = overtake, 5 = hotlap
float32 ers_harvested_this_lap_mguk  # ERS energy harvested this lap by MGU-K
float32 ers_harvested_this_lap_mguh  # ERS energy harvested this lap by MGU-H
float32 ers_deployed_this_lap       # ERS energy deployed this lap