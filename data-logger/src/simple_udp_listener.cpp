/*
Simple UDP Server
*/


#include "simple_udp_listener.h"
#include<stdio.h>
#include<winsock2.h>
#define UDP_BUFLEN 1289   //Max length of buffer
namespace deepf1
{
	simple_udp_listener::simple_udp_listener(boost::shared_ptr<const boost::timer::cpu_timer>& timer,
		unsigned int length,
		unsigned short port_number) {
		this->timer = timer;
		this->length = length;
		this->port_number = port_number;
		dataz.reserve(length);
		for (unsigned int i = 0; i < length; i++)
		{
			timestamped_udp_data_t to_add;
			to_add.data = new UDPPacket();
			dataz.push_back(to_add);
		}
	}
	simple_udp_listener::~simple_udp_listener() {
		for (unsigned int i = 0; i < dataz.size(); i++)
		{
			delete dataz[i].data;
		}
	}
	std::vector<timestamped_udp_data_t> simple_udp_listener::get_data() {
		return dataz;
	}
	void simple_udp_listener::listen()
	{
		SOCKET s;
		struct sockaddr_in server, si_other;
		int slen, recv_len;
		WSADATA wsa;

		slen = sizeof(si_other);

		//Initialise winsock
		printf("\nInitialising Winsock...");
		if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
		{
			printf("Failed. Error Code : %d", WSAGetLastError());
			exit(EXIT_FAILURE);
		}
		printf("Initialised.\n");

		//Create a socket
		if ((s = socket(AF_INET, SOCK_DGRAM, 0)) == INVALID_SOCKET)
		{
			printf("Could not create socket : %d", WSAGetLastError());
		}
		printf("Socket created.\n");

		//Prepare the sockaddr_in structure
		server.sin_family = AF_INET;
		server.sin_addr.s_addr = INADDR_ANY;
		server.sin_port = htons(port_number);

		//Bind
		if (bind(s, (struct sockaddr *)&server, sizeof(server)) == SOCKET_ERROR)
		{
			printf("Bind failed with error code : %d", WSAGetLastError());
			exit(EXIT_FAILURE);
		}
		//keep listening for data
		unsigned int i = 0;
		while (i++ < length)
		{


			int rcv_len = recvfrom(s, (char*)(dataz[i].data), UDP_BUFLEN, 0, (struct sockaddr *) &si_other, &slen);
			if (rcv_len != UDP_BUFLEN) {
				printf("Socket error when receiving telemetry data.");
				exit(-1);
			}
			dataz[i].timestamp = timer->elapsed();
			//printf("Steering angle: %f\n", packet->m_steer);

		}

		closesocket(s);
		WSACleanup();
	}
}