#include "f1_datalogger/udp_logging/common/rebroadcast_handler_2018.h"
#include <boost/bind.hpp>
#include <iostream>
deepf1::RebroadcastHandler2018::RebroadcastHandler2018()
{
	std::cout << "Constructing Rebroadcast Handler" << std::endl;
}
deepf1::RebroadcastHandler2018::~RebroadcastHandler2018()
{
}
void handle_send(const std::string metadata,
  const boost::system::error_code& error,
  std::size_t bytes_transferred)
{
  if(error.failed())
  {
   std::printf("Failed to rebroadcast %lu bytes. Error code: %d. Error Message: %s\n",
              bytes_transferred, error.value(), error.message().c_str());
   std::printf("Message metadata: %s\n", metadata.c_str());
  }
}
std::string getMetadata(const deepf1::twenty_eighteen::PacketHeader& header)
{
	return "Frame id: " + std::to_string(header.m_frameIdentifier) +"\n"+
        "Packet Format: " + std::to_string(header.m_packetFormat) +"\n"+
        "Packet id: " + std::to_string(header.m_packetId) +"\n"+
        "Packet version: " + std::to_string(header.m_packetVersion) +"\n"+
        "Player Car Index: " + std::to_string(header.m_playerCarIndex) +"\n"+
        "Session Time: " + std::to_string(header.m_sessionTime) +"\n"+
        "Session UID: " + std::to_string(header.m_sessionUID) +"\n";
}
void deepf1::RebroadcastHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data)
{
	std::string metadata= getMetadata(data.data.m_header);
	socket_->async_send_to(boost::asio::buffer(&(data.data), sizeof(data.data)), *remote_endpoint_, 0,
	boost::bind(&handle_send, metadata,
		boost::asio::placeholders::error,
		boost::asio::placeholders::bytes_transferred));
	 io_service_->run_one();

}
void deepf1::RebroadcastHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data)
{
	std::string metadata= getMetadata(data.data.m_header);
	 socket_->async_send_to(boost::asio::buffer(&(data.data), sizeof(data.data)), *remote_endpoint_, 0,
	boost::bind(&handle_send, metadata,
		boost::asio::placeholders::error,
		boost::asio::placeholders::bytes_transferred));
	 io_service_->run_one();
}
void deepf1::RebroadcastHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data)
{
	std::string metadata= getMetadata(data.data.m_header);
	 socket_->async_send_to(boost::asio::buffer(&(data.data), sizeof(data.data)), *remote_endpoint_, 0,
	boost::bind(&handle_send, metadata,
		boost::asio::placeholders::error,
		boost::asio::placeholders::bytes_transferred));
	 io_service_->run_one();
		
}
void deepf1::RebroadcastHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data)
{
	std::string metadata= getMetadata(data.data.m_header);
	 socket_->async_send_to(boost::asio::buffer(&(data.data), sizeof(data.data)), *remote_endpoint_, 0,
	boost::bind(&handle_send, metadata,
		boost::asio::placeholders::error,
		boost::asio::placeholders::bytes_transferred));
	 io_service_->run_one();

}
void deepf1::RebroadcastHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data)
{
	std::string metadata= getMetadata(data.data.m_header);
	 socket_->async_send_to(boost::asio::buffer(&(data.data), sizeof(data.data)), *remote_endpoint_, 0,
	boost::bind(&handle_send, metadata,
		boost::asio::placeholders::error,
		boost::asio::placeholders::bytes_transferred));
	 io_service_->run_one();

}
void deepf1::RebroadcastHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data)
{
	std::string metadata= getMetadata(data.data.m_header);
	 socket_->async_send_to(boost::asio::buffer(&(data.data), sizeof(data.data)), *remote_endpoint_, 0,
	boost::bind(&handle_send, metadata,
		boost::asio::placeholders::error,
		boost::asio::placeholders::bytes_transferred));
	 io_service_->run_one();

}
void deepf1::RebroadcastHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data)
{
	std::string metadata= getMetadata(data.data.m_header);
	 socket_->async_send_to(boost::asio::buffer(&(data.data), sizeof(data.data)), *remote_endpoint_, 0,
	boost::bind(&handle_send, metadata,
		boost::asio::placeholders::error,
		boost::asio::placeholders::bytes_transferred));
	 io_service_->run_one();

}
void deepf1::RebroadcastHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data)
{
	std::string metadata= getMetadata(data.data.m_header);
	 socket_->async_send_to(boost::asio::buffer(&(data.data), sizeof(data.data)), *remote_endpoint_, 0,
	boost::bind(&handle_send, metadata,
		boost::asio::placeholders::error,
		boost::asio::placeholders::bytes_transferred));
	 io_service_->run_one();

}
bool deepf1::RebroadcastHandler2018::isReady()
{
	return bool(remote_endpoint_) && bool(socket_) && bool(io_service_);
}
void deepf1::RebroadcastHandler2018::init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin)
{
  io_service_.reset(new boost::asio::io_service);
  socket_.reset(new boost::asio::ip::udp::socket(*io_service_));
  socket_->open(boost::asio::ip::udp::v4());
  remote_endpoint_.reset(new boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(host), port + 1));
}
