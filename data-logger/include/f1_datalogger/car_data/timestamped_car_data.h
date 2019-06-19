#ifndef TIMESTAMPED_CAR_DATA_H
#define  TIMESTAMPED_CAR_DATA_H


#include "f1_datalogger/car_data/car_data.h"
#include <chrono>
namespace deepf1
{
	typedef std::chrono::high_resolution_clock::time_point TimePoint;
	struct timestamped_udp_data {
		UDPPacket2017 data;
		TimePoint timestamp;
	}; 
	typedef struct timestamped_udp_data TimestampedUDPData;
/*
  virtual void handleData(const deepf1::twenty_eighteen::PacketCarSetupData& data) = 0;
  virtual void handleData(const deepf1::twenty_eighteen::PacketCarStatusData& data) = 0;
  virtual void handleData(const deepf1::twenty_eighteen::PacketCarTelemetryData& data) = 0;
  virtual void handleData(const deepf1::twenty_eighteen::PacketEventData& data) = 0;
  virtual void handleData(const deepf1::twenty_eighteen::PacketLapData& data) = 0;
  virtual void handleData(const deepf1::twenty_eighteen::PacketMotionData& data) = 0;
  virtual void handleData(const deepf1::twenty_eighteen::PacketParticipantsData& data) = 0;
  virtual void handleData(const deepf1::twenty_eighteen::PacketSessionData& data) = 0;
*/
namespace twenty_eighteen
{
	struct TimestampedPacketCarSetupData{
		TimestampedPacketCarSetupData(const PacketCarSetupData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		deepf1::twenty_eighteen::PacketCarSetupData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketCarStatusData{
		TimestampedPacketCarStatusData(const PacketCarStatusData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		deepf1::twenty_eighteen::PacketCarStatusData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketCarTelemetryData{
		TimestampedPacketCarTelemetryData(const PacketCarTelemetryData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		deepf1::twenty_eighteen::PacketCarTelemetryData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketEventData{
		TimestampedPacketEventData(const PacketEventData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		deepf1::twenty_eighteen::PacketEventData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketLapData{
		TimestampedPacketLapData(const PacketLapData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		deepf1::twenty_eighteen::PacketLapData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketMotionData{
		TimestampedPacketMotionData(const PacketMotionData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		deepf1::twenty_eighteen::PacketMotionData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketParticipantsData{
		TimestampedPacketParticipantsData(const PacketParticipantsData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		deepf1::twenty_eighteen::PacketParticipantsData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketSessionData{
		TimestampedPacketSessionData(const PacketSessionData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		deepf1::twenty_eighteen::PacketSessionData data;
		TimePoint timestamp;
	};
}
	
}

#endif
