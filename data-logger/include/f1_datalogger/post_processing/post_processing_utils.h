
#ifndef INCLUDE_F1_DATALOGGER_POST_PROCESSING_UTILS_H_
#define INCLUDE_F1_DATALOGGER_POST_PROCESSING_UTILS_H_
#include <vector>
#include "f1_datalogger/proto/TimestampedUDPData.pb.h"
#include "f1_datalogger/proto/TimestampedImage.pb.h"
#include "f1_datalogger/proto/LabeledImage.pb.h"
namespace deepf1 
{
namespace post_processing
{
	class PostProcessingUtils
	{
	public:
		PostProcessingUtils();
		virtual ~PostProcessingUtils();

		static std::vector<deepf1::protobuf::TimestampedUDPData> parseUDPDirectory(const std::string& directory);
		static std::vector<deepf1::protobuf::TimestampedImage> parseImageDirectory(const std::string& directory);
		static std::vector<deepf1::protobuf::LabeledImage> labelImages(std::vector<deepf1::protobuf::TimestampedUDPData>& udp_data, std::vector<deepf1::protobuf::TimestampedImage>& image_data, unsigned int interpolation_order = 3);
		static std::pair<deepf1::protobuf::TimestampedUDPData, unsigned int> closestValue(const std::vector<deepf1::protobuf::TimestampedUDPData>& sorted_data, google::protobuf::uint64 search);


	};
}
}


#endif

