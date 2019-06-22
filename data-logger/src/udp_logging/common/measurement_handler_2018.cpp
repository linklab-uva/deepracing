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

const deepf1::twenty_eighteen::TimestampedPacketCarSetupData deepf1::MeasurementHandler2018::getCurrentSetupData() const
{
	return current_setup_data_;
}
const deepf1::twenty_eighteen::TimestampedPacketCarStatusData deepf1::MeasurementHandler2018::getCurrentStatusData() const
{
	return current_status_data_;
}
const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData deepf1::MeasurementHandler2018::getCurrentTelemetryData() const
{
	return current_telemetry_data_;
}
const deepf1::twenty_eighteen::TimestampedPacketEventData deepf1::MeasurementHandler2018::getCurrentEventData() const
{
	return current_event_data_;
}
const deepf1::twenty_eighteen::TimestampedPacketLapData deepf1::MeasurementHandler2018::getCurrentLapData() const
{
	return current_lap_data_;
}
const deepf1::twenty_eighteen::TimestampedPacketMotionData deepf1::MeasurementHandler2018::getCurrentMotionData() const
{
	return current_motion_data_;
}
const deepf1::twenty_eighteen::TimestampedPacketParticipantsData deepf1::MeasurementHandler2018::getCurrentParticipantData() const
{
	return current_participant_data_;
}
const deepf1::twenty_eighteen::TimestampedPacketSessionData deepf1::MeasurementHandler2018::getCurrentSessionData() const
{
	return current_session_data_;
}


void deepf1::MeasurementHandler2018::init(const std::string& host, unsigned int port, const std::chrono::high_resolution_clock::time_point& begin)
{

}
