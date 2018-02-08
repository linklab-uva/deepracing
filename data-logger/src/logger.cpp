
#include <Windows.h>
#include <stdio.h>
#include <boost/timer/timer.hpp>
#include "simple_udp_listener.h"
#include "simple_screen_listener.h"
#include <iostream>
#include <exception>
#include <thread>
#include <string>
#include <functional>
#define BUFLEN 1289   //Max length of buffer
#define PORT 20777   //The port on which to listen for incoming data
#define DEFAULT_MAX_UDP_FRAMES 2000
#define DEFAULT_MAX_IMAGE_FRAMES 10
int main(int argc, char** argv) {

	boost::shared_ptr<const boost::timer::cpu_timer> timer(new boost::timer::cpu_timer);
	boost::shared_ptr<const boost::timer::cpu_timer> timer2(new boost::timer::cpu_timer);
	unsigned int udp_len, image_len;
	if (argc > 1) {
		udp_len = atoi(argv[1]);
	}
	else {
		udp_len = DEFAULT_MAX_UDP_FRAMES;
	}
	if (argc > 2) {
		image_len = atoi(argv[2]);
	}
	else {
		image_len = DEFAULT_MAX_IMAGE_FRAMES;
	}
	/*
	deepf1::simple_screen_listener screen_listener(timer2);
	screen_listener.listen();
	*/
	deepf1::simple_udp_listener udp_listener(timer, udp_len);
	std::function<void ()> udp_worker = std::bind(&deepf1::simple_udp_listener::listen, &udp_listener);
	std::thread udp_thread(udp_worker);

	deepf1::simple_screen_listener screen_listener(timer2);
	std::function<void()> screen_worker = std::bind(&deepf1::simple_screen_listener::listen, &screen_listener);
	std::thread screen_thread(screen_worker);

	std::cin.get();
	screen_thread.join();
	udp_thread.join();
	return 0;
//	printf("Value is: %f", udp_listener.get_data()[10].data.m_steer);
	//boost::function<void (const boost::timer::cpu_timer&, timestamped_udp_data_t[], unsigned int)> f1 = boost::bind(&udp_worker, (const boost::timer::cpu_timer) timer, udp_dataz, MAX_FRAMES);
	//
	//boost::thread screencap_trhead(boost::bind(&::screencap_worker, image_dataz, timer, MAX_FRAMES));

	return 0;
}