#pragma once
#include "car_data/timestamped_car_data.h"
#include "car_data/timestamped_image_data.h"
#include <deepf1_gsoap/deepf1_gsoap.nsmap>
namespace deepf1{
	class gsoap_conversions
	{
	public:
		gsoap_conversions();
		~gsoap_conversions();
		static deepf1_gsoap::UDPPacket* convert_to_gsoap(const deepf1::UDPPacket& udp_data, soap* soap);
	};
}
