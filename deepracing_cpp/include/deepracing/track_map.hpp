#ifndef DEEPRACING_CPP__TRACK_MAP_HPP_
#define DEEPRACING_CPP__TRACK_MAP_HPP_

#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <deepracing/pcl_types.hpp>
#include <deepracing/visibility_control.hpp>
#include <Eigen/Geometry>
#include <unordered_map>

namespace deepracing
{
    class TrackMap
    {
      
        public:
            DEEPRACING_CPP_PUBLIC TrackMap();
            void DEEPRACING_CPP_PUBLIC loadFromDirectory(const std::string& track_directory);
            pcl::PCLPointCloud2 DEEPRACING_CPP_PUBLIC getCloud(const std::string& key);
            const pcl::PointCloud<PointXYZLapdistance>::ConstPtr DEEPRACING_CPP_PUBLIC innerBound();
            const pcl::PointCloud<PointXYZLapdistance>::ConstPtr DEEPRACING_CPP_PUBLIC outerBound();
            const pcl::PointCloud<PointXYZTime>::ConstPtr DEEPRACING_CPP_PUBLIC raceline();
            const std::string DEEPRACING_CPP_PUBLIC name();

        private:
            pcl::PointCloud<PointXYZLapdistance>::ConstPtr inner_boundary_, outer_boundary_;
            pcl::PointCloud<PointXYZTime>::ConstPtr raceline_;
            double startinglinewidth_, tracklength_;
            Eigen::Isometry3d startingline_pose_;          
            std::unordered_map<std::string, pcl::PCLPointCloud2> other_clouds_;
            std::string name_;
    };
    TrackMap DEEPRACING_CPP_PUBLIC findTrackmap(const std::string& trackname, const std::vector<std::string> & search_dirs);
}

#endif