#include "deepf1_gsoap_conversions/gsoap_conversions.h"


namespace deepf1 {
	gsoap_conversions::gsoap_conversions()
	{
	}


	gsoap_conversions::~gsoap_conversions()
	{
	}
	deepf1_gsoap::UDPPacket* convert_to_gsoap(const deepf1::UDPPacket& udp_data, soap* soap) {
		deepf1_gsoap::UDPPacket* rtn = deepf1_gsoap::soap_new_UDPPacket(soap);
		rtn->m_ang_acc_x = udp_data.m_ang_acc_x;
		rtn->m_ang_acc_y = udp_data.m_ang_acc_y;
		rtn->m_ang_acc_z = udp_data.m_ang_acc_z;
		rtn->m_ang_vel_x = udp_data.m_ang_vel_x;
		rtn->m_ang_vel_y = udp_data.m_ang_vel_y;
		rtn->m_ang_vel_z = udp_data.m_ang_vel_z;
		rtn->m_anti_lock_brakes = udp_data.m_anti_lock_brakes;
		rtn->m_brake = udp_data.m_brake;
		for (unsigned int i = 0; i < 4; i++) {
			rtn->m_brakes_temp[i] = udp_data.m_brakes_temp[i];
		}

		return rtn;
	}
}