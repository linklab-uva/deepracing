#include "f1_datalogger/udp_logging/common/measurement_handler_2018.h"
deepf1::MeasurementHandler2018::MeasurementHandler2018()
{
}
deepf1::MeasurementHandler2018::~MeasurementHandler2018()
{
}
void deepf1::MeasurementHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data)
{
	this->current_setup_data_ = data;
}
void deepf1::MeasurementHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data)
{
	this->current_status_data_ = data;
}
void deepf1::MeasurementHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data)
{
	this->current_telemetry_data_ = data;
}
void deepf1::MeasurementHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data)
{
	this->current_event_data_ = data;
}
void deepf1::MeasurementHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data)
{
	this->current_lap_data_ = data;
}
void deepf1::MeasurementHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data)
{
	this->current_motion_data_ = data;
}
void deepf1::MeasurementHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data)
{
	this->current_participant_data_ = data;
}
void deepf1::MeasurementHandler2018::handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data)
{
	this->current_session_data_ = data;
}
bool deepf1::MeasurementHandler2018::isReady()
{
	return true;
}

void deepf1::MeasurementHandler2018::init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin)
{

}
