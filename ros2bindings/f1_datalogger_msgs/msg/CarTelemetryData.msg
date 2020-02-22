uint16 speed                      # Speed of car in kilometres per hour
uint8 throttle                   # Amount of throttle applied (0 to 100)
int8 steer                      # Steering (-100 (full lock left) to 100 (full lock right))
uint8 brake                      # Amount of brake applied (0 to 100)
uint8 clutch                     # Amount of clutch applied (0 to 100)
int8 gear                       # Gear selected (1-8, N=0, R=-1)
uint16 engine_rpm                  # Engine RPM
uint8 drs                       # 0 = off, 1 = on
uint8 rev_lights_percent           # Rev lights indicator (percentage)
uint16[4] brakes_temperature       # Brakes temperature (celsius)
uint16[4] tyres_surface_temperature # Tyres surface temperature (celsius)
uint16[4] tyres_inner_temperature   # Tyres inner temperature (celsius)
uint16 engine_temperature          # Engine temperature (celsius)
float32[4] tyres_pressure           # Tyres pressure (PSI)