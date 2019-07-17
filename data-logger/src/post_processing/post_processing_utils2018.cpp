#include "f1_datalogger/post_processing/post_processing_utils2018.h"
#include <google/protobuf/util/json_util.h>
#include "f1_datalogger/alglib/interpolation.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <Eigen/Geometry>
#if defined(__GNUC__) && (__GNUC__ < 8)
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret)
{
  if (!fs::exists(root) || !fs::is_directory(root)) return;

  fs::recursive_directory_iterator it(root);
  fs::recursive_directory_iterator endit;

  while (it != endit)
  {
    if (fs::is_regular_file(*it) && it->path().extension() == ext) ret.push_back(it->path().filename());
    ++it;
  }

}

std::vector<deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData> 
deepf1::post_processing::PostProcessingUtils2018::parseMotionPacketDirectory(const std::string& directory, bool json)
{

  std::vector<deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData> rtn;
  std::vector<fs::path> paths;
  std::string ext("pb");
  if (json)
  {
    ext = "json";
  }
  get_all( fs::path(directory), ext, paths );
  for each (const fs::path & current_path in paths)
  {
    deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData packet;
    std::ifstream stream_in(current_path.string());
    if (json)
    {
      std::stringstream buffer;
      buffer << stream_in.rdbuf();
      google::protobuf::util::Status rc = google::protobuf::util::JsonStringToMessage(buffer.str(), &packet);
      if ( !rc.ok() )
      {
        std::printf("Could not load JSON file: %s\n", current_path.string().c_str());
      }
    }
    else
    {
      if (!packet.ParseFromIstream(&stream_in))
      {
        std::printf("Could not load binary file: %s\n", current_path.string().c_str());
      }
    }
    rtn.push_back(packet);
  }


  return rtn;

}