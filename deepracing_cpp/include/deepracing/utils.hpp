#ifndef DEEPRACING_CPP__UTILS_H_
#define DEEPRACING_CPP__UTILS_H_
#include <map>
#include <string>
#include <deepracing/pcl_types.hpp>
#include <pcl/point_cloud.h>

namespace deepracing
{
    class Utils
    {
        public:
            static std::map<std::int8_t, std::string> trackNames();
            static pcl::PointCloud<deepracing::PointXYZLapdistance> closeBoundary(const pcl::PointCloud<deepracing::PointXYZLapdistance>& open_boundary);
    };
    // inline std::map<std::int8_t, std::string> trackNames();
}



#endif