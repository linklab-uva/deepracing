#include <deepracing/track_map.hpp>
#include <deepracing/pcl_transforms.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <sstream>
#include <algorithm>
#include <cctype>


namespace fs = std::filesystem;
namespace deepracing
{  
    TrackMap::TrackMap(const std::string& name, 
                const pcl::PointCloud<PointXYZLapdistance>& innerbound, 
                const pcl::PointCloud<PointXYZLapdistance>& outerbound, 
                const pcl::PointCloud<PointXYZTime>& raceline)
    {
        name_=name;
        inner_boundary_.reset(new pcl::PointCloud<PointXYZLapdistance>(innerbound));
        outer_boundary_.reset(new pcl::PointCloud<PointXYZLapdistance>(outerbound));
        raceline_.reset(new pcl::PointCloud<PointXYZTime>(raceline));
    }
    TrackMap::TrackMap()
    {

    }
    TrackMap::Ptr TrackMap::getptr()
    {
        return shared_from_this();
    }
    TrackMap::Ptr TrackMap::create(const std::string& name, 
                const pcl::PointCloud<PointXYZLapdistance>& innerbound, 
                const pcl::PointCloud<PointXYZLapdistance>& outerbound, 
                const pcl::PointCloud<PointXYZTime>& raceline)
    {
        return TrackMap::Ptr(new TrackMap(name, innerbound, outerbound, raceline));
    }
    void TrackMap::loadFromDirectory(const std::string& track_directory)
    {
        fs::path track_directory_path(track_directory); 
        fs::path metadata_path = track_directory_path/fs::path("metadata.yaml");
        YAML::Node track_config = YAML::LoadFile(metadata_path.string()); 
        name_ = track_config["name"].as<std::string>(); 
        tracklength_ = track_config["tracklength"].as<double>(); 
        startinglinewidth_ = track_config["startinglinewidth"].as<double>(); 
        YAML::Node startingline_pose_node = track_config["startingline_pose"]; 
        std::vector<double> startinglineposition = startingline_pose_node["position"].as<std::vector<double>>();
        if(!(startinglineposition.size()==3))
        {
            throw std::runtime_error("\"position\" key in metadata.yaml must be a list of 3 floats");
        }
        std::vector<double> startinglinequaternion = startingline_pose_node["quaternion"].as<std::vector<double>>();
        if(!(startinglinequaternion.size()==4))
        {
            throw std::runtime_error("\"quaternion\" key in metadata.yaml must be a list of 4 floats");
        }

        Eigen::Vector3d peigen = Eigen::Map<Eigen::Vector3d>(startinglineposition.data(), startinglineposition.size());
        Eigen::Quaterniond qeigen = Eigen::Quaterniond(startinglinequaternion.at(3), startinglinequaternion.at(0), startinglinequaternion.at(1), startinglinequaternion.at(2));
        startingline_pose_.fromPositionOrientationScale(peigen, qeigen, Eigen::Vector3d::Ones());
        Eigen::Isometry3f pcl_transform = startingline_pose_.inverse().cast<float>();   

        fs::path inner_boundary_file_path = track_directory_path / fs::path("inner_boundary.pcd");
        pcl::PointCloud<PointXYZLapdistance> temp, temp_map;
        if(!(pcl::io::loadPCDFile<PointXYZLapdistance>(inner_boundary_file_path.string(), temp)==0))
        {
            std::stringstream errorstream;
            errorstream<<"Could not load inner boundary file: " << inner_boundary_file_path.string() << std::endl;
            throw std::runtime_error(errorstream.str());
        }
        pcl::transformPointCloud<PointXYZLapdistance>(temp, temp_map, pcl_transform);
        temp_map.header.frame_id="map";
        inner_boundary_.reset(new pcl::PointCloud<PointXYZLapdistance>(temp_map));

        temp.clear();
        temp_map.clear();
        fs::path outer_boundary_file_path = track_directory_path / fs::path("outer_boundary.pcd");
        if(!(pcl::io::loadPCDFile<PointXYZLapdistance>(outer_boundary_file_path.string(), temp)==0))
        {
            std::stringstream errorstream;
            errorstream<<"Could not load outer boundary file: " << outer_boundary_file_path.string() << std::endl;
            throw std::runtime_error(errorstream.str());
        }
        pcl::transformPointCloud<PointXYZLapdistance>(temp, temp_map, pcl_transform);
        temp_map.header.frame_id="map";
        outer_boundary_.reset(new pcl::PointCloud<PointXYZLapdistance>(temp_map));

        pcl::PointCloud<PointXYZTime> temp_raceline, temp_raceline_map;
        fs::path raceline_file_path = track_directory_path / fs::path("raceline.pcd");
        if(!(pcl::io::loadPCDFile<PointXYZTime>(raceline_file_path.string(), temp_raceline)==0))
        {
            std::stringstream errorstream;
            errorstream<<"Could not load raceline file: " << raceline_file_path.string() << std::endl;
            throw std::runtime_error(errorstream.str());
        }
        pcl::transformPointCloud<PointXYZTime>(temp_raceline, temp_raceline_map, pcl_transform);
        temp_raceline_map.header.frame_id="map";
        raceline_.reset(new pcl::PointCloud<PointXYZTime>(temp_raceline_map));

        fs::path widthmap_filepath = track_directory_path / fs::path("widthmap.pcd");
        pcl::PointCloud<PointWidthMap> widthmaptemp, widthmaptemp_map;
        if(pcl::io::loadPCDFile<PointWidthMap>(widthmap_filepath.string(), widthmaptemp)==0)
        {
            deepracing::transforms::transformPointCloudWithOrientation<PointWidthMap>(widthmaptemp, widthmaptemp_map, pcl_transform);
            widthmaptemp_map.header.frame_id="map";
            width_map_.reset(new pcl::PointCloud<PointWidthMap>(widthmaptemp_map));
        }

        other_clouds_.clear();
        for (const fs::directory_entry& dir_entry : fs::recursive_directory_iterator(track_directory_path))
        {
            const fs::path& current_filepath = dir_entry.path();
            const std::string& key = current_filepath.stem().string();
            if (!dir_entry.is_regular_file() || 
                key=="inner_boundary" || 
                key=="outer_boundary" || 
                key=="widthmap"|| 
                key=="raceline" )
            {
                continue;
            }
            std::string extension(current_filepath.extension().string());
            std::string extension_lowercase(extension.size(), 0);
            std::transform(extension.begin(), extension.end(), extension_lowercase.begin(),
                [](unsigned char c){ return std::tolower(c); });
            if(extension_lowercase==".pcd")
            {
                pcl::PCLPointCloud2 pc2;
                if(pcl::io::loadPCDFile(current_filepath.string(), pc2)==0)
                {
                    pc2.header.frame_id="track";
                    other_clouds_[key] = pc2;
                }
            }
        }

    }
    
    const std::vector<std::string> TrackMap::keys() const
    {
        std::vector<std::string> rtn;
        for (auto it = other_clouds_.begin(); it!=other_clouds_.end(); it++)
        {
            rtn.push_back(it->first);
        }
        return rtn;
    }
    const pcl::PCLPointCloud2 TrackMap::getCloud(const std::string& key) const
    {
        pcl::PCLPointCloud2 rtn;
        try
        {
            rtn = other_clouds_.at(key);
            return rtn;
        }
        catch(const std::out_of_range& e)
        {
            std::cerr << "Key " << key << " doesn't exist in this trackmap" << std::endl;
            throw e;
        }
        catch(const std::exception& e)
        {
            std::cerr << "Unknown error when loading trackfile " << key << std::endl;
            std::cerr << "Underlying exception:"<< std::endl << std::string(e.what()) << std::endl;
            throw e;
        }
    }
    const pcl::PointCloud<PointWidthMap>::ConstPtr TrackMap::widthMap() const
    {
        return width_map_;
    }
    const pcl::PointCloud<PointXYZLapdistance>::ConstPtr TrackMap::innerBound() const
    {
        return inner_boundary_;
    }
    const pcl::PointCloud<PointXYZLapdistance>::ConstPtr TrackMap::outerBound() const
    {
        return outer_boundary_;
    }
    const pcl::PointCloud<PointXYZTime>::ConstPtr TrackMap::raceline() const
    {
        return raceline_;
    }
    const std::string TrackMap::name() const
    {
        return name_;
    }
    const Eigen::Isometry3d TrackMap::startinglinePose() const
    {
        return startingline_pose_;
    }
    TrackMap::Ptr TrackMap::findTrackmap(const std::string& trackname, const std::vector<std::string> & search_dirs)
    {
        TrackMap::Ptr rtn;
        for(const std::string& search_dir : search_dirs)
        {
            fs::path search_path(search_dir);
            for(const fs::directory_entry& dir_entry : fs::recursive_directory_iterator(search_path))
            {
                if(dir_entry.exists() && dir_entry.is_directory())
                {
                    fs::path checkpath = dir_entry.path();
                    fs::path metadata_checkpath = checkpath/fs::path("metadata.yaml");
                    fs::path marker_checkpath = checkpath/fs::path("DEEPRACING_TRACKMAP");
                    if( fs::exists(checkpath) && 
                        fs::is_directory(checkpath) && 
                        fs::exists(metadata_checkpath) && 
                        fs::is_regular_file(metadata_checkpath) && 
                        fs::exists(marker_checkpath) && 
                        fs::is_regular_file(marker_checkpath))
                    {
                        YAML::Node track_config = YAML::LoadFile(metadata_checkpath.string()); 
                        std::cout<<track_config["name"]<<", " << trackname << std::endl;
                        if(track_config["name"])
                        {
                            if(track_config["name"].as<std::string>()==trackname)
                            {
                                rtn.reset(new TrackMap);
                                rtn->loadFromDirectory(checkpath.string());
                                return rtn;
                            }
                        }
                        else
                        {
                            std::cerr << "Found a directory " << checkpath.string() << " with both metadata.yaml and DEEPRACING_TRACKMAP, but metadata.yaml does not have a \"name\" key" << std::endl;
                        }
                    }
                }
            }
        }
        return rtn;
        // std::stringstream errorstream;
        // errorstream << "Could not find map for track " << trackname << std::endl;
        // throw std::runtime_error(errorstream.str());
    }
}


