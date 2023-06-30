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
    class TrackMap : public std::enable_shared_from_this<TrackMap>
    {
      
        public:
            typedef std::shared_ptr<TrackMap> Ptr;
            typedef std::shared_ptr<const TrackMap> ConstPtr;
            void DEEPRACING_CPP_PUBLIC loadFromDirectory(const std::string& track_directory);
            pcl::PCLPointCloud2 DEEPRACING_CPP_PUBLIC getCloud(const std::string& key);
            const Eigen::Isometry3d DEEPRACING_CPP_PUBLIC startinglinePose();
            const pcl::PointCloud<PointXYZLapdistance>::ConstPtr DEEPRACING_CPP_PUBLIC innerBound();
            const pcl::PointCloud<PointXYZLapdistance>::ConstPtr DEEPRACING_CPP_PUBLIC outerBound();
            const pcl::PointCloud<PointXYZTime>::ConstPtr DEEPRACING_CPP_PUBLIC raceline();
            const std::string DEEPRACING_CPP_PUBLIC name();
            TrackMap::Ptr DEEPRACING_CPP_PUBLIC getptr();
            [[nodiscard]] static TrackMap::Ptr DEEPRACING_CPP_PUBLIC create(const std::string& name, 
                const pcl::PointCloud<PointXYZLapdistance>& innerbound, 
                const pcl::PointCloud<PointXYZLapdistance>& outerbound, 
                const pcl::PointCloud<PointXYZTime>& raceline);
            static TrackMap::Ptr DEEPRACING_CPP_PUBLIC findTrackmap(const std::string& trackname, const std::vector<std::string> & search_dirs);
        private:
            pcl::PointCloud<PointXYZLapdistance>::ConstPtr inner_boundary_, outer_boundary_;
            pcl::PointCloud<PointXYZTime>::ConstPtr raceline_;
            double startinglinewidth_, tracklength_;
            Eigen::Isometry3d startingline_pose_;          
            std::unordered_map<std::string, pcl::PCLPointCloud2> other_clouds_;
            std::string name_;
            DEEPRACING_CPP_PUBLIC TrackMap();
            DEEPRACING_CPP_PUBLIC TrackMap(const std::string& name, 
                const pcl::PointCloud<PointXYZLapdistance>& innerbound, 
                const pcl::PointCloud<PointXYZLapdistance>& outerbound, 
                const pcl::PointCloud<PointXYZTime>& raceline);
    };
}

#endif