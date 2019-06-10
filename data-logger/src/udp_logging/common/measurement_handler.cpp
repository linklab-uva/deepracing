#include "f1_datalogger/udp_logging/common/measurement_handler.h"
deepf1::MeasurementHandler::MeasurementHandler()
{
		
}


deepf1::MeasurementHandler::~MeasurementHandler()
{
}

deepf1::TimestampedUDPData deepf1::MeasurementHandler::getData() const
{
	return this->data_;
}

void deepf1::MeasurementHandler::handleData(const deepf1::TimestampedUDPData& data)
{
	this->data_ = data;
}

bool deepf1::MeasurementHandler::isReady()
{
	return true;
}

void deepf1::MeasurementHandler::init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin)
{

}
