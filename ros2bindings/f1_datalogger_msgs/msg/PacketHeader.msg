uint16 packet_format# 2018
uint8 packet_version# Version of this packet type, all start from 1
uint8 packet_id# Identifier for the packet type, see below
uint64 session_uid# Unique identifier for the session
float32 session_time# Session timestamp
uint32 frame_identifier# Identifier for the frame the data was retrieved on
uint8 player_car_index# Index of player's car in the array