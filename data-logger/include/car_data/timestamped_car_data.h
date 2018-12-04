#pragma once
#include "car_data/car_data.h"
#include <chrono>
namespace deepf1
{
	struct timestamped_udp_data {
		UDPPacket data;
		std::chrono::microseconds timestamp;
	}; 
	typedef struct timestamped_udp_data timestamped_udp_data_t;
}