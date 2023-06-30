#include <deepracing/track_map.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <sstream>
#include <algorithm>
#include <cctype>


namespace fs = std::filesystem;
namespace deepracing
{  
    TrackMap::TrackMap()
    {

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
        std::vector<double> startinglinequaternion = startingline_pose_node["quaternion"].as<std::vector<double>>();
        if(!(startinglinequaternion.size()==4))
        {
            throw std::runtime_error("\"quaternion\" key in metadata.yaml must be a list of 4 floats");
        }

        Eigen::Vector3d peigen = Eigen::Map<Eigen::Vector3d>(startinglineposition.data(), startinglineposition.size());
        Eigen::Quaterniond qeigen = Eigen::Quaterniond(startinglinequaternion.back(), startinglinequaternion.at(0), startinglinequaternion.at(1), startinglinequaternion.at(2));
        startingline_pose_.fromPositionOrientationScale(peigen, qeigen, Eigen::Vector3d::Ones());
        Eigen::Isometry3d startingline_pose_inverse = startingline_pose_.inverse();   
        fs::path inner_boundary_file_path = track_directory_path / fs::path("inner_boundary.pcd");
        
        pcl::PointCloud<PointXYZLapdistance> temp;
        if(!(pcl::io::loadPCDFile<PointXYZLapdistance>(inner_boundary_file_path.string(), temp)==0))
        {
            std::stringstream errorstream;
            errorstream<<"Could not load inner boundary file: " << inner_boundary_file_path.string() << std::endl;
            throw std::runtime_error(errorstream.str());
        }
        inner_boundary_.reset(new pcl::PointCloud<PointXYZLapdistance>(temp));

        temp.clear();
        fs::path outer_boundary_file_path = track_directory_path / fs::path("outer_boundary.pcd");
        if(!(pcl::io::loadPCDFile<PointXYZLapdistance>(outer_boundary_file_path.string(), temp)==0))
        {
            std::stringstream errorstream;
            errorstream<<"Could not load outer boundary file: " << outer_boundary_file_path.string() << std::endl;
            throw std::runtime_error(errorstream.str());
        }
        outer_boundary_.reset(new pcl::PointCloud<PointXYZLapdistance>(temp));

        pcl::PointCloud<PointXYZTime> temp_raceline;
        fs::path raceline_file_path = track_directory_path / fs::path("raceline.pcd");
        if(!(pcl::io::loadPCDFile<PointXYZTime>(raceline_file_path.string(), temp_raceline)==0))
        {
            std::stringstream errorstream;
            errorstream<<"Could not load raceline file: " << raceline_file_path.string() << std::endl;
            throw std::runtime_error(errorstream.str());
        }
        raceline_.reset(new pcl::PointCloud<PointXYZTime>(temp_raceline));

        for (const fs::directory_entry& dir_entry : fs::recursive_directory_iterator(track_directory_path))
        {
            const fs::path& current_filepath = dir_entry.path();
            const std::string& key = current_filepath.stem().string();
            if (!dir_entry.is_regular_file() || 
                key=="inner_boundary" || 
                key=="outer_boundary" || 
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
                    other_clouds_[key] = pc2;
                }
            }
        }

    }
    
    pcl::PCLPointCloud2 TrackMap::getCloud(const std::string& key)
    {
        pcl::PCLPointCloud2 rtn;
        try
        {
            rtn = other_clouds_.at(key);
            return rtn;
        }
        catch(const std::out_of_range& e)
        {
            std::stringstream errorstream;
            errorstream << "Key " << key << " doesn't exist in this trackmap" << std::endl;
            throw std::runtime_error(errorstream.str());
        }
        catch(const std::exception& e)
        {
            std::stringstream errorstream;
            errorstream << "Unknown error when loading trackfile " << key << std::endl;
            errorstream << "Underlying exception:"<< std::endl << std::string(e.what()) << std::endl;
            throw std::runtime_error(errorstream.str());
        }
        
    }
    const pcl::PointCloud<PointXYZLapdistance>::ConstPtr TrackMap::innerBound()
    {
        return inner_boundary_;
    }
    const pcl::PointCloud<PointXYZLapdistance>::ConstPtr TrackMap::outerBound()
    {
        return outer_boundary_;
    }
    const pcl::PointCloud<PointXYZTime>::ConstPtr TrackMap::raceline()
    {
        return raceline_;
    }
    const std::string TrackMap::name()
    {
        return name_;
    }
    TrackMap findTrackmap(const std::string& trackname, const std::vector<std::string> & search_dirs)
    {
        for(const std::string& search_dir : search_dirs)
        {
            fs::path search_path(search_dir);
            for(const fs::directory_entry& dir_entry : fs::recursive_directory_iterator(search_path))
            {
                if(dir_entry.is_directory())
                {
                    fs::path checkpath = dir_entry.path()/fs::path(trackname);
                    fs::path metadata_checkpath = checkpath/fs::path("metadata.yaml");
                    fs::path marker_checkpath = checkpath/fs::path("DEEPRACING_TRACKMAP");
                    if( fs::exists(checkpath) && 
                        fs::is_directory(checkpath) && 
                        fs::exists(metadata_checkpath) && 
                        fs::is_regular_file(metadata_checkpath) && 
                        fs::exists(marker_checkpath) && 
                        fs::is_regular_file(marker_checkpath))
                    {
                        TrackMap rtn;
                        rtn.loadFromDirectory(checkpath.string());
                        return rtn;
                    }
                }
            }
        }
        std::stringstream errorstream;
        errorstream << "Could not find map for track " << trackname << std::endl;
        throw std::runtime_error(errorstream.str());
    }
}


