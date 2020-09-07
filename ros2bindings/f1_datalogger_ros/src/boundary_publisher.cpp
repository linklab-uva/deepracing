#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <fstream>
#include <streambuf>
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
    
    std::array<uint32_t,3> sizes = {(uint32_t)xarray.size(), (uint32_t)yarray.size(), (uint32_t)zarray.size()};
    if (! std::all_of(sizes.begin(), sizes.end(), [xarray, yarray, zarray](uint32_t i){return i==xarray.size();}) )
    {
       RCLCPP_FATAL(node->get_logger(), "All three arrays are not the same size. Size of x array: %lu. Size of y array: %lu. Size of z array: %lu",sizes[0], sizes[1], sizes[2]); 
    }
    return rootval;
}

void unpackDictionary(std::shared_ptr<rclcpp::Node> node, const Json::Value& boundary_dict, pcl::PointCloud<pcl::PointXYZINormal>& cloudpcl, f1_datalogger_msgs::msg::BoundaryLine& blmsg, geometry_msgs::msg::PoseArray& pose_array, Eigen::Vector3d ref = Eigen::Vector3d::UnitY() )
{
    Json::Value boundary_x = boundary_dict["x"];Json::Value boundary_y = boundary_dict["y"];Json::Value boundary_z = boundary_dict["z"];
    Json::Value boundary_xtangent = boundary_dict["x_tangent"];Json::Value boundary_ytangent = boundary_dict["y_tangent"];Json::Value boundary_ztangent = boundary_dict["z_tangent"];
    Json::Value boundary_dist = boundary_dict["dist"];
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
    if(  boundary_dist.isNull() )
    {
        RCLCPP_FATAL(node->get_logger(), "no \"dist\" key found in the json dictionary");
    }
    cloudpcl.clear();
    blmsg.x.clear();
    blmsg.y.clear();
    blmsg.z.clear();
    blmsg.xtangent.clear();
    blmsg.ytangent.clear();
    blmsg.ztangent.clear();
    blmsg.dist.clear();
    pose_array.poses.clear();
    unsigned int imax = boundary_x.size();
    for (unsigned int i =0; i < imax; i++)
    {
        double dist = boundary_dist[i].asDouble();
        Eigen::Vector3d tangent(boundary_xtangent[i].asDouble(), boundary_ytangent[i].asDouble(), boundary_ztangent[i].asDouble());
       // tangent.normalize();
        Eigen::Vector3d normal = ref.cross(tangent);
        normal.normalize();
        pcl::PointXYZ point(boundary_x[i].asDouble(), boundary_y[i].asDouble(), boundary_z[i].asDouble());
        Eigen::Vector3d pointeig(point.x,point.y,point.z);
        unsigned int iforward = (i+5)%imax;
        Eigen::Vector3d pointforward(boundary_x[iforward].asDouble(), boundary_y[iforward].asDouble(), boundary_z[iforward].asDouble());
        Eigen::Vector3d delta = pointforward - pointeig;
        delta.normalize();
        Eigen::Vector3d refcomp = normal.cross(delta);
        if (refcomp.dot(ref)>0.0)
        {
            RCLCPP_INFO(node->get_logger(), "Flipping point %d: (%f, %f, %f)", i, point.x, point.y, point.z);
            normal *=-1.0;
        }


        pcl::PointXYZINormal pointnormal(point.x, point.y, point.z, dist, normal.x(), normal.y(), normal.z());
        cloudpcl.push_back(pointnormal);
        blmsg.x.push_back(point.x);
        blmsg.y.push_back(point.y);
        blmsg.z.push_back(point.z);
        blmsg.xtangent.push_back(tangent.x());
        blmsg.ytangent.push_back(tangent.y());
        blmsg.ztangent.push_back(tangent.z());
        blmsg.dist.push_back(dist);

        Eigen::Matrix3d rotmat;
        rotmat.col(0) = normal;
        rotmat.col(1) = normal.cross(ref);
        rotmat.col(2) = rotmat.col(0).cross(rotmat.col(1));

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

        // if(i>0)
        // {
        //     Eigen::Vector3d delta = Eigen::Vector3d(cloudpcl.at(i).x, cloudpcl.at(i).y, cloudpcl.at(i).z) - Eigen::Vector3d(cloudpcl.at(i-1).x, cloudpcl.at(i-1).y, cloudpcl.at(i-1).z);
        //     Eigen::Vector3d nprev = Eigen::Vector3d(cloudpcl.at(i-1).normal_x, cloudpcl.at(i-1).normal_y, cloudpcl.at(i-1).normal_z);
        //     Eigen::Vector3d ncurr = Eigen::Vector3d(cloudpcl.at(i).normal_x, cloudpcl.at(i).normal_y, cloudpcl.at(i).normal_z);
        //     double dot = ncurr.dot(nprev);
        //     RCLCPP_DEBUG(node->get_logger(), "Dot product between point %d and point %d: %f", i-1, i, dot);
        //     if (dot < 0.0)
        //     {
        //         RCLCPP_ERROR(node->get_logger(), "Abrubt change in normal vector detected at point (X,Y,Z): (%f,%f,%f)", point.x, point.y, point.z);
        //     }
        // }
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
    
    std::shared_ptr< rclcpp::Publisher<f1_datalogger_msgs::msg::BoundaryLine> > innerpubbl = node->create_publisher<f1_datalogger_msgs::msg::BoundaryLine>("/inner_track_boundary",1);
    std::shared_ptr< rclcpp::Publisher<f1_datalogger_msgs::msg::BoundaryLine> > outerpubbl = node->create_publisher<f1_datalogger_msgs::msg::BoundaryLine>("/outer_track_boundary",1);
    std::shared_ptr< rclcpp::Publisher<f1_datalogger_msgs::msg::BoundaryLine> > racelinepubbl = node->create_publisher<f1_datalogger_msgs::msg::BoundaryLine>("/optimal_raceline",1);

    
    std::shared_ptr< rclcpp::Publisher<geometry_msgs::msg::PoseArray> > innerpubpa = node->create_publisher<geometry_msgs::msg::PoseArray>("/inner_track_boundary/pose_array",1);
    std::shared_ptr< rclcpp::Publisher<geometry_msgs::msg::PoseArray> > outerpubpa = node->create_publisher<geometry_msgs::msg::PoseArray>("/outer_track_boundary/pose_array",1);
    std::shared_ptr< rclcpp::Publisher<geometry_msgs::msg::PoseArray> > racelinepubpa = node->create_publisher<geometry_msgs::msg::PoseArray>("/optimal_raceline/pose_array",1);
    
    NodeWrapper_ nw(node);
    pcl::PointCloud<pcl::PointXYZINormal> innercloudPCL, outercloudPCL, racelinecloudPCL;
    sensor_msgs::msg::PointCloud2 innercloudMSG, outercloudMSG, racelinecloudMSG;
    Json::Value innerDict, outerDict, racelineDict;
    f1_datalogger_msgs::msg::BoundaryLine innerBL, outerBL, racelineBL;
    geometry_msgs::msg::PoseArray innerPA, outerPA, racelinePA;

    rclcpp::ParameterValue track_dir_param = node->declare_parameter("track_dir",rclcpp::ParameterValue(""));
    std::string track_dir = track_dir_param.get<std::string>();
    std::string track_name = "";
    std::array<std::string,25> track_name_array = f1_datalogger_ros::F1MsgUtils::track_names();

    while (rclcpp::is_initialized())
    {
        rclcpp::sleep_for(std::chrono::nanoseconds(int(1E9)));
        rclcpp::spin_some(node);
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
            innerBL.track_name = track_name;
            outerBL.track_name = track_name;
            racelineBL.track_name = track_name;

            std::string inner_filename = track_name + "_innerlimit.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", inner_filename.c_str(), track_dir.c_str());
            innerDict = readJsonFile(node,(fs::path(track_dir) / fs::path(inner_filename)).string());
            unpackDictionary(node,innerDict, innercloudPCL, innerBL, innerPA, -Eigen::Vector3d::UnitY());
            RCLCPP_INFO(node->get_logger(), "Got %d points for the inner boundary.", innercloudPCL.size());

            std::string outer_filename = track_name + "_outerlimit.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", outer_filename.c_str(), track_dir.c_str());
            outerDict = readJsonFile(node,(fs::path(track_dir) / fs::path(outer_filename)).string());
            unpackDictionary(node,outerDict,outercloudPCL,outerBL,outerPA);
            RCLCPP_INFO(node->get_logger(), "Got %d points for the outer boundary.", outercloudPCL.size());

            std::string raceline_filename = track_name + "_racingline.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", raceline_filename.c_str(), track_dir.c_str());
            racelineDict = readJsonFile(node,(fs::path(track_dir) / fs::path(raceline_filename)).string());
            unpackDictionary(node,racelineDict,racelinecloudPCL,racelineBL,racelinePA);
            RCLCPP_INFO(node->get_logger(), "Got %d points for the optimal raceline.", racelinecloudPCL.size());

            pcl::PCLPointCloud2 innercloudPC2, outercloudPC2, racelinecloudPC2;
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZINormal>(innercloudPCL), innercloudPC2);
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZINormal>(outercloudPCL), outercloudPC2);
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZINormal>(racelinecloudPCL), racelinecloudPC2);
            pcl_conversions::moveFromPCL(innercloudPC2, innercloudMSG);
            pcl_conversions::moveFromPCL(outercloudPC2, outercloudMSG);
            pcl_conversions::moveFromPCL(racelinecloudPC2, racelinecloudMSG);
            // outercloudMSG.header.frame_id = f1_datalogger_ros::F1MsgUtils::world_coordinate_name; 
            // racelinecloudMSG.header.frame_id = f1_datalogger_ros::F1MsgUtils::world_coordinate_name; 
            // innerBL.header.frame_id = f1_datalogger_ros::F1MsgUtils::world_coordinate_name; 
            // outerBL.header.frame_id = f1_datalogger_ros::F1MsgUtils::world_coordinate_name; 
            // racelineBL.header.frame_id = f1_datalogger_ros::F1MsgUtils::world_coordinate_name; 


        }
        innercloudMSG.header.frame_id = f1_datalogger_ros::F1MsgUtils::world_coordinate_name; 
        innercloudMSG.header.stamp = node->now();
        outercloudMSG.header = innercloudMSG.header;
        racelinecloudMSG.header = innercloudMSG.header;
        innerBL.header = innercloudMSG.header;
        outerBL.header = innercloudMSG.header;
        racelineBL.header = innercloudMSG.header;
        innerPA.header = innercloudMSG.header;
        outerPA.header = innercloudMSG.header;
        racelinePA.header = innercloudMSG.header;

        innerpub->publish(innercloudMSG);
        outerpub->publish(outercloudMSG);
        racelinepub->publish(racelinecloudMSG);
        
        innerpubbl->publish(innerBL);
        outerpubbl->publish(outerBL);
        racelinepubbl->publish(racelineBL);
        
        innerpubpa->publish(innerPA);
        outerpubpa->publish(outerPA);
        racelinepubpa->publish(racelinePA);


    }


    


}