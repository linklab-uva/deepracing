
#include <Windows.h>
#include <stdio.h>
#include <boost/timer/timer.hpp>
#include "simple_udp_listener.h"
#include <iostream>
#include <exception>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
#include <string>
#define BUFLEN 1289   //Max length of buffer
#define PORT 20777   //The port on which to listen for incoming data
#define DEFAULT_MAX_UDP_FRAMES 500
int main(int argc, char** argv) {

	boost::shared_ptr<const boost::timer::cpu_timer> timer(new boost::timer::cpu_timer);
	unsigned int len;
	if (argc > 1) {
		len = atoi(argv[1]);
	}
	else {
		len = DEFAULT_MAX_UDP_FRAMES;
	}
	simple_udp_listener udp_listener(timer, len);
	boost::thread udp_thread(boost::bind(&simple_udp_listener::listen, &udp_listener));
	//udp_thread.join();
	std::string s;
	std::getline(std::cin, s);
	printf("Value is: %f", udp_listener.get_data()[10].data.m_steer);
	//boost::function<void (const boost::timer::cpu_timer&, timestamped_udp_data_t[], unsigned int)> f1 = boost::bind(&udp_worker, (const boost::timer::cpu_timer) timer, udp_dataz, MAX_FRAMES);
	//
	//boost::thread screencap_trhead(boost::bind(&::screencap_worker, image_dataz, timer, MAX_FRAMES));

	return 0;
}