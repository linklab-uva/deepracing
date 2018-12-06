/*
 * simulated_telemetry_server.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */



#include "car_data/car_data.h"
#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <memory>
#include <boost/program_options.hpp>
#include <thread>
using boost::asio::ip::udp;
using namespace deepf1;
namespace po = boost::program_options;
int main(int argc, char** argv) {
        unsigned int BUFLEN = 1289;
        unsigned int UDP_BUFLEN = BUFLEN;
        unsigned int sleep_time;
        unsigned int packet_size = sizeof(UDPPacket);
        std::string address, port;
        po::options_description desc("Allowed Options");
        desc.add_options()
                ("help,h", "Displays options and exits")
                ("address,a", po::value<std::string>(&address)->default_value("127.0.0.1"), "IPv4 Address to send data to")
                ("port_number,p", po::value<std::string>(&port)->default_value("20777"), "Port number to send data to")
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
                exit(0); // @suppress("Invalid arguments")
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
        float dt = (float)sleep_time;
        while (true) {
                std::cout<<"Sending fake UDP data"<<std::endl;
                data->m_time = fake_time;
                socket.send_to(boost::asio::buffer(boost::asio::buffer(data.get(), packet_size)), receiver_endpoint);
                fake_time += dt;
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        }
        return 0;
}
