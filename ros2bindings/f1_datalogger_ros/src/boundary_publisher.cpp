#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "rclcpp/rclcpp.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <json/json.h>
#include <f1_datalogger_msgs/msg/timestamped_packet_session_data.hpp>
#include <f1_datalogger_msgs/msg/boundary_line.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include "f1_datalogger_ros/utils/f1_msg_utils.h"

Json::Value readJsonFile(std::shared_ptr<rclcpp::Node> node, std::string filepath)
{
    Json::Value rootval;
    std::ifstream file;
    file.open(filepath);
    RCLCPP_INFO(node->get_logger(), "Reading json");
    std::string json;
    file.seekg(0, std::ios::end);   
    json.reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    json.assign((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    file.close();

    RCLCPP_INFO(node->get_logger(), "Parsing json");
    Json::Reader reader;
    if(reader.parse(json,rootval))
    {
        RCLCPP_INFO(node->get_logger(), "Parsed json");
    }
    else
    {
        RCLCPP_FATAL(node->get_logger(), "Failed to parse json");
    }
    Json::Value xarray = rootval["x"];
    if(  xarray.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"x\" key found in the json dictionary");
    }
    Json::Value yarray = rootval["y"];
    if(  yarray.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"y\" key found in the json dictionary");
    }
    Json::Value zarray = rootval["z"];
    if(  zarray.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"z\" key found in the json dictionary");
    }
    Json::Value boundary_xtangent = rootval["x_tangent"];Json::Value boundary_ytangent = rootval["y_tangent"];Json::Value boundary_ztangent = rootval["z_tangent"];
    Json::Value boundary_xnormal = rootval["x_normal"];Json::Value boundary_ynormal = rootval["y_normal"];Json::Value boundary_znormal = rootval["z_normal"];
    Json::Value boundary_dist = rootval["dist"];
    if(  boundary_xtangent.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"x_tangent\" key found in the json dictionary");
    }
    if(  boundary_ytangent.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"y_tangent\" key found in the json dictionary");
    }
    if(  boundary_ztangent.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"z_tangent\" key found in the json dictionary");
    }
    if(  boundary_xnormal.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"x_normal\" key found in the json dictionary");
    }
    if(  boundary_ynormal.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"y_normal\" key found in the json dictionary");
    }
    if(  boundary_znormal.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"z_normal\" key found in the json dictionary");
    }
    if(  boundary_dist.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"dist\" key found in the json dictionary");
    }
    
    std::array<uint32_t,10> sizes = {(uint32_t)xarray.size(), (uint32_t)yarray.size(), (uint32_t)zarray.size(),
                                    (uint32_t)xarray.size(), (uint32_t)yarray.size(), (uint32_t)zarray.size(),
                                    (uint32_t)xarray.size(), (uint32_t)yarray.size(), (uint32_t)zarray.size(), (uint32_t)boundary_dist.size()};
    if (! std::all_of(sizes.begin(), sizes.end(), [xarray](uint32_t i){return i==xarray.size();}) )
    {
       std::stringstream ss;
       std::for_each(sizes.begin(), sizes.end(), [&ss](uint32_t i){ss<<i<<std::endl;});
       RCLCPP_FATAL(node->get_logger(), "All arrays are not the same size. Sizes: %s", ss.str().c_str()); 
    }
    return rootval;
}

void unpackDictionary(std::shared_ptr<rclcpp::Node> node, const Json::Value& boundary_dict, pcl::PointCloud<pcl::PointXYZINormal>& cloudpcl, geometry_msgs::msg::PoseArray& pose_array, Eigen::Vector3d ref = Eigen::Vector3d::UnitY() )
{
    Json::Value boundary_x = boundary_dict["x"];Json::Value boundary_y = boundary_dict["y"];Json::Value boundary_z = boundary_dict["z"];
    Json::Value boundary_xtangent = boundary_dict["x_tangent"];Json::Value boundary_ytangent = boundary_dict["y_tangent"];Json::Value boundary_ztangent = boundary_dict["z_tangent"];
    Json::Value boundary_xnormal = boundary_dict["x_normal"];Json::Value boundary_ynormal = boundary_dict["y_normal"];Json::Value boundary_znormal = boundary_dict["z_normal"];
    Json::Value boundary_dist = boundary_dict["dist"];
    cloudpcl.clear();
    pose_array.poses.clear();
    unsigned int imax = boundary_x.size();
    for (unsigned int i =0; i < imax; i++)
    {
        double dist = boundary_dist[i].asDouble();
        Eigen::Vector3d tangent(boundary_xtangent[i].asDouble(), boundary_ytangent[i].asDouble(), boundary_ztangent[i].asDouble());
        tangent.normalize();
        Eigen::Vector3d normal(boundary_xnormal[i].asDouble(), boundary_ynormal[i].asDouble(), boundary_znormal[i].asDouble());
        normal.normalize();
       
        pcl::PointXYZ point(boundary_x[i].asDouble(), boundary_y[i].asDouble(), boundary_z[i].asDouble());
        pcl::PointXYZINormal pointnormal(point.x, point.y, point.z, dist, normal.x(), normal.y(), normal.z());
        cloudpcl.push_back(pointnormal);

        Eigen::Matrix3d rotmat;
        rotmat.col(0) = normal;
        rotmat.col(2) = tangent;
        Eigen::Vector3d yvec = rotmat.col(2).cross(rotmat.col(0));
        yvec.normalize();
        rotmat.col(1)=yvec;
        Eigen::Quaterniond quat(rotmat);


        geometry_msgs::msg::Pose pose;
        pose.position.x = point.x;
        pose.position.y = point.y;
        pose.position.z = point.z;
        pose.orientation.x = quat.x();
        pose.orientation.y = quat.y();
        pose.orientation.z = quat.z();
        pose.orientation.w = quat.w();
        pose_array.poses.push_back(pose);
    }
}
class NodeWrapper_
{
    public: 
        NodeWrapper_(std::shared_ptr<rclcpp::Node> node)
        {
            if(!bool(node))
            {
                this->node = rclcpp::Node::make_shared("f1_boundary_publisher","");
            }
            else
            {
                this->node=node;
            }
            listener =  this->node->create_subscription<f1_datalogger_msgs::msg::TimestampedPacketSessionData>( "/session_data", 1, std::bind(&NodeWrapper_::sessionDataCallback, this, std::placeholders::_1)  );
            current_session_data.reset(new f1_datalogger_msgs::msg::TimestampedPacketSessionData);
            current_session_data->udp_packet.track_id=-1;
        }
        std::shared_ptr<f1_datalogger_msgs::msg::TimestampedPacketSessionData> current_session_data;
    private:
        std::shared_ptr<rclcpp::Node> node;
        std::shared_ptr< rclcpp::Subscription<f1_datalogger_msgs::msg::TimestampedPacketSessionData> > listener;
        void sessionDataCallback(const f1_datalogger_msgs::msg::TimestampedPacketSessionData::SharedPtr session_data_packet)
        {
            this->current_session_data.reset(new f1_datalogger_msgs::msg::TimestampedPacketSessionData(*session_data_packet));
        }

};
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    namespace fs = std::filesystem;
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("f1_boundary_publisher","");
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > innerpub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/inner_track_boundary/pcl",1);
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > outerpub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/outer_track_boundary/pcl",1);
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > racelinepub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/optimal_raceline/pcl",1);

    
    std::shared_ptr< rclcpp::Publisher<geometry_msgs::msg::PoseArray> > innerpubpa = node->create_publisher<geometry_msgs::msg::PoseArray>("/inner_track_boundary/pose_array",1);
    std::shared_ptr< rclcpp::Publisher<geometry_msgs::msg::PoseArray> > outerpubpa = node->create_publisher<geometry_msgs::msg::PoseArray>("/outer_track_boundary/pose_array",1);
    std::shared_ptr< rclcpp::Publisher<geometry_msgs::msg::PoseArray> > racelinepubpa = node->create_publisher<geometry_msgs::msg::PoseArray>("/optimal_raceline/pose_array",1);
    
    NodeWrapper_ nw(node);
    pcl::PointCloud<pcl::PointXYZINormal> innercloudPCL, outercloudPCL, racelinecloudPCL;
    sensor_msgs::msg::PointCloud2 innercloudMSG, outercloudMSG, racelinecloudMSG;
    Json::Value innerDict, outerDict, racelineDict;
    geometry_msgs::msg::PoseArray innerPA, outerPA, racelinePA;

    rclcpp::ParameterValue track_dir_param = node->declare_parameter("track_dir",rclcpp::ParameterValue(std::string(std::getenv("F1_TRACK_DIR"))));
    std::string track_dir = track_dir_param.get<std::string>();
    std::string track_name = "";
    std::array<std::string,25> track_name_array = f1_datalogger_ros::F1MsgUtils::track_names();

    while (rclcpp::is_initialized())
    {
        rclcpp::sleep_for(std::chrono::nanoseconds(int(1E9)));
        rclcpp::spin_some(node);
        innercloudMSG.header.stamp = node->now();
        RCLCPP_DEBUG(node->get_logger(), "Ran a spin.");
        std::string active_track="";
        int8_t track_index = nw.current_session_data->udp_packet.track_id;
        if(track_index>=0 && track_index<track_name_array.size())
        {
            active_track=track_name_array[track_index];
        }
        RCLCPP_DEBUG(node->get_logger(), "current_track: %s", track_name.c_str());
        RCLCPP_DEBUG(node->get_logger(), "active_track: %s", active_track.c_str());
        if(!active_track.empty() && track_name.compare(active_track)!=0)
        {            
            track_name = active_track;
            RCLCPP_INFO(node->get_logger(), "Detected new track  %s.", track_name.c_str());

            std::string inner_filename = track_name + "_innerlimit.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", inner_filename.c_str(), track_dir.c_str());
            innerDict = readJsonFile(node,(fs::path(track_dir) / fs::path(inner_filename)).string());
            unpackDictionary(node,innerDict, innercloudPCL, innerPA, -Eigen::Vector3d::UnitY());
            RCLCPP_INFO(node->get_logger(), "Got %d points for the inner boundary.", innercloudPCL.size());

            std::string outer_filename = track_name + "_outerlimit.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", outer_filename.c_str(), track_dir.c_str());
            outerDict = readJsonFile(node,(fs::path(track_dir) / fs::path(outer_filename)).string());
            unpackDictionary(node,outerDict,outercloudPCL,outerPA);
            RCLCPP_INFO(node->get_logger(), "Got %d points for the outer boundary.", outercloudPCL.size());

            std::string raceline_filename = track_name + "_racingline.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", raceline_filename.c_str(), track_dir.c_str());
            racelineDict = readJsonFile(node,(fs::path(track_dir) / fs::path(raceline_filename)).string());
            unpackDictionary(node,racelineDict,racelinecloudPCL,racelinePA);
            RCLCPP_INFO(node->get_logger(), "Got %d points for the optimal raceline.", racelinecloudPCL.size());

            pcl::PCLPointCloud2 innercloudPC2, outercloudPC2, racelinecloudPC2;
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZINormal>(innercloudPCL), innercloudPC2);
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZINormal>(outercloudPCL), outercloudPC2);
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZINormal>(racelinecloudPCL), racelinecloudPC2);
            pcl_conversions::moveFromPCL(innercloudPC2, innercloudMSG);
            pcl_conversions::moveFromPCL(outercloudPC2, outercloudMSG);
            pcl_conversions::moveFromPCL(racelinecloudPC2, racelinecloudMSG);
        }
        innercloudMSG.header.frame_id = f1_datalogger_ros::F1MsgUtils::world_coordinate_name; 
        outercloudMSG.set__header( innercloudMSG.header );
        racelinecloudMSG.set__header( innercloudMSG.header );
        innerPA.set__header( innercloudMSG.header );
        outerPA.set__header( innercloudMSG.header );
        racelinePA.set__header( innercloudMSG.header );

        innerpub->publish(innercloudMSG);
        outerpub->publish(outercloudMSG);
        racelinepub->publish(racelinecloudMSG);
        
        innerpubpa->publish(innerPA);
        outerpubpa->publish(outerPA);
        racelinepubpa->publish(racelinePA);


    }


    


}