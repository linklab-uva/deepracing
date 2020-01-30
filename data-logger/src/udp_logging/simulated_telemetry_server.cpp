/*
 * simulated_telemetry_server.cpp
 *
 *  Created on: Dec 5, 2018
 *      Author: ttw2xk
 */



#include "f1_datalogger/car_data/timestamped_car_data.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <boost/asio.hpp>
#include <memory>
#include <thread>
#include <math.h> 
#include <boost/math/constants/constants.hpp>
#include <boost/math_fwd.hpp>
namespace po = boost::program_options;
void exit_with_help(po::options_description& desc)
{
        std::stringstream ss;
        ss << "F1 Simulated Telemetry Server. Command line arguments are as follows:" << std::endl;
        desc.print(ss);
        std::printf("%s", ss.str().c_str());
        exit(0); // @suppress("Invalid arguments")
}
int main(int argc, char** argv) {
		using boost::asio::ip::udp;
		using namespace deepf1;
		using namespace deepf1::twenty_eighteen;
	unsigned int packet_size = sizeof(deepf1::twenty_eighteen::PacketMotionData);// BUFLEN;
        unsigned int sleep_time;

        std::string address, port;
        po::options_description desc("Allowed Options");
	
        try{
                desc.add_options()
                ("help,h", "Displays options and exits")
                ("address,a", po::value<std::string>(&address)->default_value("127.0.0.1"), "IPv4 Address to send data to")
                ("port_number,p", po::value<std::string>(&port)->default_value("20777"), "Port number to send data to")
                ("sleep_time,s", po::value<unsigned int>(&sleep_time)->default_value(17), "Number of milliseconds to sleep between simulated packets")
                ;
	        po::variables_map vm;
                po::store(po::parse_command_line(argc, argv, desc), vm);
                po::notify(vm);
                if (vm.find("help") != vm.end()) {
                        exit_with_help(desc);
                }
        }catch(boost::exception& e){
                exit_with_help(desc);
        }
        boost::asio::io_service io_service;
        udp::resolver resolver(io_service);
        udp::resolver::query query(udp::v4(), address, port);
        udp::endpoint receiver_endpoint = *resolver.resolve(query);
        udp::socket socket(io_service);
        socket.open(udp::v4());


        std::shared_ptr<PacketCarTelemetryData> data(new PacketCarTelemetryData);
        data->m_header.m_packetFormat=2;
        data->m_header.m_packetId=PacketID::CARTELEMETRY;
        data->m_header.m_packetVersion=18;
        data->m_header.m_playerCarIndex=0;
        float fake_time = 0;
        float dt = 1E-3*( (float) sleep_time );
        float period = 1.0;
        float freq=1/period;
	float pi = boost::math::constants::pi<float>();
	float twopi = boost::math::constants::two_pi<float>();
        float factor = 2.0;
        int id=0;
        while (true) {
                data->m_header.m_frameIdentifier=id;
                data->m_header.m_sessionTime=fake_time;
                data->m_header.m_sessionUID=id;
                data->m_carTelemetryData[data->m_header.m_playerCarIndex].m_steer = (int8_t)(100.0*std::sin(twopi*freq*fake_time));
                data->m_carTelemetryData[data->m_header.m_playerCarIndex].m_throttle = 50 + (int8_t)(50.0*std::cos(twopi*freq*fake_time));
                data->m_carTelemetryData[data->m_header.m_playerCarIndex].m_brake = 50;
                //std::cout<<"Sending fake UDP data"<<std::endl;
                // std::cout<<"fake_time: "<<fake_time<<std::endl;
                // std::cout<<"Steering: "<<data->m_steer<<std::endl;
                // std::cout<<"Throttle: "<<data->m_throttle<<std::endl;
                // std::cout<<"B:rake "<<data->m_brake<<std::endl;
                socket.send_to(boost::asio::buffer(boost::asio::buffer(data.get(), packet_size)), receiver_endpoint);
                fake_time += dt;
                id++;
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        }
        return 0;
		/* */
}
