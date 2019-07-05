#include "f1_datalogger/post_processing/post_processing_utils2018.h"
#include <filesystem>
#include <google/protobuf/util/json_util.h>
#include "f1_datalogger/alglib/interpolation.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <Eigen/Geometry>
namespace fs = std::filesystem;
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
  get_all(fs::path(directory), ext, paths);
  for each (const fs::path & path in paths)
  {
    deepf1::twenty_eighteen::protobuf::TimestampedPacketMotionData packet;
    if (json)
    {
      
    }
    else
    {

    }
  }


  return rtn;

}