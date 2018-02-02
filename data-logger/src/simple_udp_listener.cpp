/*
Simple UDP Server
*/

#include<stdio.h>
#include<winsock2.h>

#include "car_data\car_data.h"
#pragma comment(lib,"ws2_32.lib") //Winsock Library

#define BUFLEN sizeof(UDPPacket)  //Max length of buffer
#define PORT 20777   //The port on which to listen for incoming data
#define MAX_FRAMES 10000
int main()
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
	server.sin_port = htons(PORT);

	//Bind
	if (bind(s, (struct sockaddr *)&server, sizeof(server)) == SOCKET_ERROR)
	{
		printf("Bind failed with error code : %d", WSAGetLastError());
		exit(EXIT_FAILURE);
	}
	puts("Bind done");
	int BYTES_TO_GET = BUFLEN;
	//keep listening for data
	UDPPacket* packet = (UDPPacket*)malloc(BYTES_TO_GET);
	unsigned int i = 0;
	while (i++<MAX_FRAMES)
	{


		int rcv_len = recvfrom(s, (char*)packet, BYTES_TO_GET, 0, (struct sockaddr *) &si_other, &slen);
		if (rcv_len != BYTES_TO_GET) {
			printf("Socket error when receiving telemetry data.");
			exit(-1);
		}
		printf("Steering angle: %f\n", packet->m_steer);

	}

	closesocket(s);
	WSACleanup();
	free(packet);
	return 0;
}