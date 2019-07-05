
#ifndef INCLUDE_F1_DATALOGGER_POST_PROCESSING_UTILS_2018_H_
#define INCLUDE_F1_DATALOGGER_POST_PROCESSING_UTILS_2018_H_
#include <vector>
#include "f1_datalogger/proto/TimestampedPacketMotionData.pb.h"
#include <Eigen/Core>
namespace deepf1 
{
namespace post_processing
{
	class PostProcessingUtils2018
	{
	  public:
      static std::vector<deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData> 
        parseMotionPacketDirectory(const std::string& directory, bool json=true);

	};
}
}


#endif

