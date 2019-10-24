PacketHeader header # Header
CarMotionData[20] car_motion_data # Data for all cars on track
 # Extra player car ONLY data
float32[4] suspension_position # Note: All wheel arrays have the following order:
float32[4] suspension_velocity # RL, RR, FL, FR
float32[4] suspension_acceleration # RL, RR, FL, FR
float32[4] wheel_speed # Speed of each wheel
float32[4] wheel_slip # Slip ratio for each wheel

geometry_msgs/Vector3 local_velocity# Velocity in local space

geometry_msgs/Vector3 angular_velocity# Angular Velocity in global space

geometry_msgs/Vector3 angular_acceleration# Angular Acceleration in global space

float32 front_wheels_angle # Current front wheels angle in radians