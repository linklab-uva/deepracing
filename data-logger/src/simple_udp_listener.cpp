/*
Simple UDP Server
*/


#include "simple_udp_listener.h"
#include<stdio.h>
#include<winsock2.h>
#include <iostream>
#define UDP_BUFLEN 1289   //Max length of buffer
namespace deepf1
{
	simple_udp_listener::simple_udp_listener(std::shared_ptr<const boost::timer::cpu_timer> timer,
		unsigned int length,
		unsigned short port_number) {
		this->timer = timer;
		this->length = length;
		this->port_number_ = port_number;
		dataz.reserve(length);
		for (unsigned int i = 0; i < length; i++)
		{
			timestamped_udp_data_t to_add;
			to_add.data = new UDPPacket();
			dataz.push_back(to_add);
		}
	}
	simple_udp_listener::~simple_udp_listener() {
		int size = dataz.size();
		for (unsigned int i = 0; i < size; i++)
		{
			delete dataz[i].data;
		}
	}
	std::vector<timestamped_udp_data_t> simple_udp_listener::get_data() {
		return dataz;
	}
	void simple_udp_listener::listen()
	{
		running = true;
		SOCKET s;
		struct sockaddr_in server, si_other;
		int slen, recv_len;
		WSADATA wsa;

		slen = sizeof(si_other);

		//Initialise winsock
		//printf("\nInitialising Winsock...");
		if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
		{
			printf("Failed. Error Code : %d", WSAGetLastError());
			exit(EXIT_FAILURE);
		}
		//printf("Initialised.\n");

		//Create a socket
		if ((s = socket(AF_INET, SOCK_DGRAM, 0)) == INVALID_SOCKET)
		{
			printf("Could not create socket : %d", WSAGetLastError());
		}
		//printf("Socket created.\n");

		//Prepare the sockaddr_in structure
		server.sin_family = AF_INET;
		server.sin_addr.s_addr = INADDR_ANY;
		server.sin_port = htons(port_number_);

		//Bind
		if (bind(s, (struct sockaddr *)&server, sizeof(server)) == SOCKET_ERROR)
		{
			printf("Bind failed with error code : %d", WSAGetLastError());
			exit(EXIT_FAILURE);
		}
		//keep listening for data
		unsigned int i = 0;
		struct sockaddr* other = (struct sockaddr *) &si_other;
		collection_start_ = timer->elapsed();
		unsigned int max = dataz.size();
		for(; (i < max) && running; i++)
		{
			recvfrom(s, (char*)(dataz[i].data), UDP_BUFLEN, 0, other, &slen);
			dataz[i].timestamp = timer->elapsed();
		}
		for (int j = i + 1; j < max; j++) {
			delete dataz[j].data;
		}
		dataz.resize(i);
		//printf("Returning %d elements \n", dataz.size());
		closesocket(s);
		WSACleanup();
	}
}