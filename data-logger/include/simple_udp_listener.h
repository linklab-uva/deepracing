#pragma once
#include <boost/timer/timer.hpp>
#include "car_data/car_data.h"
#include <boost/shared_ptr.hpp>
#define MAX_UDP_FRAMES 500
namespace deepf1
{
	struct timestamped_udp_data {
		UDPPacket data;
		boost::timer::cpu_times timestamp;
	};
	typedef struct timestamped_udp_data timestamped_udp_data_t;
	class simple_udp_listener
	{
	public:
		simple_udp_listener(boost::shared_ptr<const boost::timer::cpu_timer>& timer, unsigned int length = 250);
		~simple_udp_listener();
		timestamped_udp_data_t* get_data();
		void listen();
	private:
		boost::shared_ptr<const boost::timer::cpu_timer> timer;
		timestamped_udp_data_t* dataz;
		unsigned int length;


	};
}

