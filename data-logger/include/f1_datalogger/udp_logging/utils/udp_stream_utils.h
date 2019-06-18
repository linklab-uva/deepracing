#include "f1_datalogger/proto/UDPData.pb.h"
#include "f1_datalogger/car_data/car_data.h"
namespace deepf1
{
	class UDPStreamUtils
	{
	public:
		UDPStreamUtils();
		~UDPStreamUtils();
		static deepf1::protobuf::UDPData toProto(const deepf1::UDPPacket2017& fromStream);
		static deepf1::protobuf::CarUDPData toProto(const deepf1::CarUDPData2017& fromStream);
	};
}