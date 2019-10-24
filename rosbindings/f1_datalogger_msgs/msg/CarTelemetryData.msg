uint32 m_speed                      # Speed of car in kilometres per hour
uint32 m_throttle                   # Amount of throttle applied (0 to 100)
int32 m_steer                      # Steering (-100 (full lock left) to 100 (full lock right))
uint32 m_brake                      # Amount of brake applied (0 to 100)
uint32 m_clutch                     # Amount of clutch applied (0 to 100)
int32 m_gear                       # Gear selected (1-8, N=0, R=-1)
uint32 m_engine_rpm                  # Engine RPM
uint32 m_drs                       # 0 = off, 1 = on
uint32 m_rev_lights_percent           # Rev lights indicator (percentage)
uint32[4] m_brakes_temperature       # Brakes temperature (celsius)
uint32[4] m_tyres_surface_temperature # Tyres surface temperature (celsius)
uint32[4] m_tyres_inner_temperature   # Tyres inner temperature (celsius)
uint32 m_engine_temperature          # Engine temperature (celsius)
float32[4] m_tyres_pressure           # Tyres pressure (PSI)