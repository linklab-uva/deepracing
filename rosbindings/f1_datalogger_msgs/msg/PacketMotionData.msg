PacketHeader header # Header
CarMotionData[] car_motion_data # Data for all cars on track
 # Extra player car ONLY data
float32[] suspension_position # Note: All wheel arrays have the following order:
float32[] suspension_velocity # RL, RR, FL, FR
float32[] suspension_acceleration # RL, RR, FL, FR
float32[] wheel_speed # Speed of each wheel
float32[] wheel_slip # Slip ratio for each wheel
float32 local_velocity_x # Velocity in local space
float32 local_velocity_y # Velocity in local space
float32 local_velocity_z # Velocity in local space
float32 angular_velocity_x # Angular velocity x-component
float32 angular_velocity_y # Angular velocity y-component
float32 angular_velocity_z # Angular velocity z-component
float32 angular_acceleration_x # Angular velocity x-component
float32 angular_acceleration_y # Angular velocity y-component
float32 angular_acceleration_z # Angular velocity z-component
float32 front_wheels_angle # Current front wheels angle in radians