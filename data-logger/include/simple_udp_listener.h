#pragma once
#include "car_data/timestamped_car_data.h"
#include <memory>
#include <chrono>
#include <vector>
#define MAX_UDP_FRAMES 1000
#define DEFAULT_PORT 20777   //The port on which to listen for incoming data
namespace deepf1
{

	class simple_udp_listener
	{
	public:
		simple_udp_listener(std::shared_ptr<const boost::timer::cpu_timer> timer, unsigned int length = MAX_UDP_FRAMES,
			unsigned short port_number = DEFAULT_PORT);
		~simple_udp_listener();
		std::vector<timestamped_udp_data_t> get_data();
		void listen();
	private:
		std::shared_ptr<const boost::timer::cpu_timer> timer;
		std::vector<timestamped_udp_data_t> dataz;
		unsigned int length;
		unsigned short port_number;


	};
}

