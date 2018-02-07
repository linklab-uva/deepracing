
#include <Windows.h>
#include<stdio.h>
#include <boost/timer/timer.hpp>
#include "simple_udp_listener.h"
#include <stdio.h>
#include <exception>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
#define BUFLEN 1289   //Max length of buffer
#define PORT 20777   //The port on which to listen for incoming data

int main(int argc, char** argv) {

	boost::shared_ptr<const boost::timer::cpu_timer> timer(new boost::timer::cpu_timer);
	printf("Elapsed time is: %d", timer->elapsed().wall);
	
	simple_udp_listener udp_listener(timer);
	boost::thread udp_thread(boost::bind(&simple_udp_listener::listen, &udp_listener));
	//boost::function<void (const boost::timer::cpu_timer&, timestamped_udp_data_t[], unsigned int)> f1 = boost::bind(&udp_worker, (const boost::timer::cpu_timer) timer, udp_dataz, MAX_FRAMES);
	//
	//boost::thread screencap_trhead(boost::bind(&::screencap_worker, image_dataz, timer, MAX_FRAMES));

	return 0;
}