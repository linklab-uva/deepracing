#pragma once
#include "car_data/car_data.h"
#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <Windows.h>
#include <memory>
#define BUFLEN 1289   //Max length of buffer
#define ADDR "127.0.0.1"   //Address to send data to
#define PORT "20777"   //The port on which to send data
#define UDP_BUFLEN 1289   //Max length of buffer
#include <boost/program_options.hpp>
using boost::asio::ip::udp;
using namespace deepf1;
namespace po = boost::program_options;
int main(int argc, char** argv) {
	unsigned int sleep_time;
	po::options_description desc("Allowed Options");
	desc.add_options()
		("sleep_time", po::value<unsigned int>(&sleep_time)->default_value(100), "Number of milliseconds to sleep between simulated packets")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	boost::asio::io_service io_service;
	udp::resolver resolver(io_service);
	udp::resolver::query query(udp::v4(), ADDR, PORT);
	udp::endpoint receiver_endpoint = *resolver.resolve(query);
	udp::socket socket(io_service);
	socket.open(udp::v4());


	std::shared_ptr<UDPPacket> data(new UDPPacket);
	data->m_steer = -0.5;
	data->m_throttle = -0.25;
	data->m_brake = 0.75;
	while (true) {
		printf("Sending data.\n");
		socket.send_to(boost::asio::buffer(boost::asio::buffer(data.get(), UDP_BUFLEN)), receiver_endpoint);
		Sleep(sleep_time);
	}
	return 0;
}