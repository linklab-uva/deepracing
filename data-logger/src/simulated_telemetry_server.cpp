#pragma once
#include "car_data/car_data.h"
#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
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
	std::string address, port;
	po::options_description desc("Allowed Options");
	desc.add_options()
		("help,h", "Displays options and exits")
		("address,a", po::value<std::string>(&address)->default_value(ADDR), "IPv4 Address to send data to")
		("port_number,p", po::value<std::string>(&port)->default_value(PORT), "Port number to send data to")
		("sleep_time,s", po::value<unsigned int>(&sleep_time)->default_value(100), "Number of milliseconds to sleep between simulated packets")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	if (vm.find("help") != vm.end()) {
		std::stringstream ss;
		ss << "F1 Simulated Telemetry Server. Command line arguments are as follows:" << std::endl;
		desc.print(ss);
		std::printf("%s", ss.str().c_str());
		exit(0);
	}
	boost::asio::io_service io_service;
	udp::resolver resolver(io_service);
	udp::resolver::query query(udp::v4(), address, port);
	udp::endpoint receiver_endpoint = *resolver.resolve(query);
	udp::socket socket(io_service);
	socket.open(udp::v4());


	std::shared_ptr<UDPPacket> data(new UDPPacket);
	data->m_steer = -0.5;
	data->m_throttle = -0.25;
	data->m_brake = 0.75;
	float fake_time = 0;
	float dt = ((float)(sleep_time)) / 1000.0;
	while (true) {
		data->m_time = fake_time;
		socket.send_to(boost::asio::buffer(boost::asio::buffer(data.get(), UDP_BUFLEN)), receiver_endpoint);
		fake_time += dt;
		Sleep(sleep_time);
	}
	return 0;
}