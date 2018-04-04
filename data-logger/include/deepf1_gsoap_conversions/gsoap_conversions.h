#pragma once
#include "car_data/timestamped_car_data.h"
#include "car_data/timestamped_image_data.h"
#include "deepf1_gsoap/deepf1_gsoapH.h"
namespace deepf1_gsoap_conversions{
	class gsoap_conversions
	{
	public:
		gsoap_conversions(soap* soap){
			this->soap = soap;
		}
		~gsoap_conversions();
		deepf1_gsoap::UDPPacket* convert_to_gsoap(const deepf1::UDPPacket& udp_data);
		deepf1_gsoap::CarUDPData* convert_to_gsoap_dynamic(const deepf1::CarUDPData& car_data);
		deepf1_gsoap::CarUDPData convert_to_gsoap(const deepf1::CarUDPData& car_data);
	private:
		soap* soap;
	};
}
