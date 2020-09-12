#ifndef F1_DATALOGGER_UDP_STREAM_UTILS_H
#define F1_DATALOGGER_UDP_STREAM_UTILS_H
#include "f1_datalogger/udp_logging/visibility_control.h"
#include "f1_datalogger/proto_dll_macro.h"
#include "f1_datalogger/proto/UDPData.pb.h"
#include "f1_datalogger/car_data/f1_2018/car_data.h"
#include "f1_datalogger/car_data/f1_2020/car_data.h"
#include "f1_datalogger/proto/PacketHeader.pb.h"
#include "f1_datalogger/proto/PacketCarStatusData.pb.h"
#include "f1_datalogger/proto/PacketCarSetupData.pb.h"
#include "f1_datalogger/proto/PacketCarTelemetryData.pb.h"
#include "f1_datalogger/proto/PacketEventData.pb.h"
#include "f1_datalogger/proto/PacketLapData.pb.h"
#include "f1_datalogger/proto/PacketMotionData.pb.h"
#include "f1_datalogger/proto/PacketParticipantsData.pb.h"
#include "f1_datalogger/proto/PacketSessionData.pb.h"

namespace deepf1
{
	class F1_DATALOGGER_UDP_LOGGING_PUBLIC UDPStreamUtils
	{
	public:
		UDPStreamUtils();
		~UDPStreamUtils();
		static deepf1::protobuf::UDPData toProto(const deepf1::UDPPacket2017& fromStream);
		static deepf1::protobuf::CarUDPData toProto(const deepf1::CarUDPData2017& fromStream);
	};
	namespace twenty_eighteen
	{
		class F1_DATALOGGER_UDP_LOGGING_PUBLIC TwentyEighteenUDPStreamUtils
		{
		public:
			TwentyEighteenUDPStreamUtils();
			~TwentyEighteenUDPStreamUtils();
			static deepf1::twenty_eighteen::protobuf::PacketCarStatusData toProto(const deepf1::twenty_eighteen::PacketCarStatusData& fromStream);
			static deepf1::twenty_eighteen::protobuf::PacketCarSetupData toProto(const deepf1::twenty_eighteen::PacketCarSetupData& fromStream);
			static deepf1::twenty_eighteen::protobuf::PacketCarTelemetryData toProto(const deepf1::twenty_eighteen::PacketCarTelemetryData& fromStream);
			static deepf1::twenty_eighteen::protobuf::PacketEventData toProto(const deepf1::twenty_eighteen::PacketEventData& fromStream);
			static deepf1::twenty_eighteen::protobuf::PacketLapData toProto(const deepf1::twenty_eighteen::PacketLapData& fromStream);
			static deepf1::twenty_eighteen::protobuf::PacketMotionData toProto(const deepf1::twenty_eighteen::PacketMotionData& fromStream);
			static deepf1::twenty_eighteen::protobuf::PacketParticipantsData toProto(const deepf1::twenty_eighteen::PacketParticipantsData& fromStream);
			static deepf1::twenty_eighteen::protobuf::PacketSessionData toProto(const deepf1::twenty_eighteen::PacketSessionData& fromStream);


			static deepf1::twenty_eighteen::protobuf::CarStatusData toProto(const deepf1::twenty_eighteen::CarStatusData& fromStream);
			static deepf1::twenty_eighteen::protobuf::CarSetupData toProto(const deepf1::twenty_eighteen::CarSetupData& fromStream);
			static deepf1::twenty_eighteen::protobuf::CarTelemetryData toProto(const deepf1::twenty_eighteen::CarTelemetryData& fromStream);
			static deepf1::twenty_eighteen::protobuf::LapData toProto(const deepf1::twenty_eighteen::LapData& fromStream);
			static deepf1::twenty_eighteen::protobuf::CarMotionData toProto(const deepf1::twenty_eighteen::CarMotionData& fromStream);
			static deepf1::twenty_eighteen::protobuf::ParticipantData toProto(const deepf1::twenty_eighteen::ParticipantData& fromStream);
			static deepf1::twenty_eighteen::protobuf::PacketHeader toProto(const deepf1::twenty_eighteen::PacketHeader& fromStream);
			static deepf1::twenty_eighteen::protobuf::MarshalZone toProto(const deepf1::twenty_eighteen::MarshalZone& fromStream);
		};
	}
}
#endif