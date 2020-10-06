#ifndef INCLUDE_F1_2019_TIMESTAMPED_CAR_DATA_H
#define INCLUDE_F1_2019_TIMESTAMPED_CAR_DATA_H


#include "f1_datalogger/car_data/f1_2019/car_data.h"
#include "f1_datalogger/car_data/time_point.h"
namespace deepf1
{
namespace twenty_nineteen
{
	struct TimestampedPacketCarSetupData{
		TimestampedPacketCarSetupData(const PacketCarSetupData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketCarSetupData() = default;
		deepf1::twenty_nineteen::PacketCarSetupData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketCarStatusData{
		TimestampedPacketCarStatusData(const PacketCarStatusData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketCarStatusData() = default;
		deepf1::twenty_nineteen::PacketCarStatusData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketCarTelemetryData{
		TimestampedPacketCarTelemetryData(const PacketCarTelemetryData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketCarTelemetryData() = default;
		deepf1::twenty_nineteen::PacketCarTelemetryData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketEventData{
		TimestampedPacketEventData(const PacketEventData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketEventData() = default;
		deepf1::twenty_nineteen::PacketEventData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketLapData{
		TimestampedPacketLapData(const PacketLapData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketLapData() = default;
		deepf1::twenty_nineteen::PacketLapData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketMotionData{
		TimestampedPacketMotionData(const PacketMotionData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketMotionData() = default;
		deepf1::twenty_nineteen::PacketMotionData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketParticipantsData{
		TimestampedPacketParticipantsData(const PacketParticipantsData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketParticipantsData() = default;
		deepf1::twenty_nineteen::PacketParticipantsData data;
		TimePoint timestamp;
	};
	struct TimestampedPacketSessionData{
		TimestampedPacketSessionData(const PacketSessionData& data, const TimePoint& timestamp)
		{
			this->data=data;
			this->timestamp=timestamp;	
		}
		TimestampedPacketSessionData () = default;
		deepf1::twenty_nineteen::PacketSessionData data;
		TimePoint timestamp;
	};
}
	
}

#endif
