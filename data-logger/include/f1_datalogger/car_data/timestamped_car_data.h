#ifndef TIMESTAMPED_CAR_DATA_H
#define TIMESTAMPED_CAR_DATA_H


#include "f1_datalogger/car_data/car_data.h"
#include "f1_datalogger/car_data/time_point.h"
namespace deepf1
{
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
	enum PacketID
	{
		MOTION=0,
		SESSION=1,
		LAPDATA=2,
		EVENT=3,
		PARTICIPANTS=4,
		CARSETUPS=5,
		CARTELEMETRY=6,
		CARSTATUS=7
	};
	struct TimestampedPacketCarSetupData{
		TimestampedPacketCarSetupData(const PacketCarSetupData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketCarSetupData() = default;
		deepf1::twenty_eighteen::PacketCarSetupData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketCarStatusData{
		TimestampedPacketCarStatusData(const PacketCarStatusData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketCarStatusData() = default;
		deepf1::twenty_eighteen::PacketCarStatusData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketCarTelemetryData{
		TimestampedPacketCarTelemetryData(const PacketCarTelemetryData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketCarTelemetryData() = default;
		deepf1::twenty_eighteen::PacketCarTelemetryData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketEventData{
		TimestampedPacketEventData(const PacketEventData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketEventData() = default;
		deepf1::twenty_eighteen::PacketEventData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketLapData{
		TimestampedPacketLapData(const PacketLapData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketLapData() = default;
		deepf1::twenty_eighteen::PacketLapData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketMotionData{
		TimestampedPacketMotionData(const PacketMotionData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketMotionData() = default;
		deepf1::twenty_eighteen::PacketMotionData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketParticipantsData{
		TimestampedPacketParticipantsData(const PacketParticipantsData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketParticipantsData() = default;
		deepf1::twenty_eighteen::PacketParticipantsData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketSessionData{
		TimestampedPacketSessionData(const PacketSessionData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketSessionData () = default;
		deepf1::twenty_eighteen::PacketSessionData data;
		TimePoint timestamp;
	};
}
	
}

#endif
