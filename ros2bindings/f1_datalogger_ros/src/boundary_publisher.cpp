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
pcl::PointCloud<pcl::PointXYZ> loadTrackFile(std::shared_ptr<rclcpp::Node> node, const std::string& filepath)
{
    Json::Value boundary_dict = readJsonFile(node, filepath);
    Json::Value boundary_x = boundary_dict["x"];
    Json::Value boundary_y = boundary_dict["y"];
    Json::Value boundary_z = boundary_dict["z"];
    pcl::PointCloud<pcl::PointXYZ> rtn;
    for (unsigned int i =0; i < boundary_x.size(); i++)
    {
        rtn.push_back(pcl::PointXYZ(boundary_x[i].asDouble(), boundary_y[i].asDouble(), boundary_z[i].asDouble()));
    }
    return rtn;
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
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > innerpub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/inner_track_boundary",1);
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > outerpub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/outer_track_boundary",1);
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > racelinepub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/optimal_raceline",1);
    
    NodeWrapper_ nw(node);
    pcl::PointCloud<pcl::PointXYZ> innercloudPCL, outercloudPCL, racelinecloudPCL;
    sensor_msgs::msg::PointCloud2 innercloudMSG, outercloudMSG, racelinecloudMSG;

    rclcpp::ParameterValue track_dir_param = node->declare_parameter("track_dir",rclcpp::ParameterValue(""));
    std::string track_dir = track_dir_param.get<std::string>();
    std::string track_name = "";
    std::array<std::string,25> track_name_array = f1_datalogger_ros::F1MsgUtils::track_names();

    while (rclcpp::is_initialized())
    {
        rclcpp::sleep_for(std::chrono::nanoseconds(1000000000));
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

            std::string inner_filename = track_name + "_innerlimit.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", inner_filename.c_str(), track_dir.c_str());
            innercloudPCL = loadTrackFile(node,(fs::path(track_dir) / fs::path(inner_filename)).string());

            std::string outer_filename = track_name + "_outerlimit.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", outer_filename.c_str(), track_dir.c_str());
            outercloudPCL = loadTrackFile(node,(fs::path(track_dir) / fs::path(outer_filename)).string());
            RCLCPP_INFO(node->get_logger(), "Got %d points for the outer boundary.", outercloudPCL.size());

            std::string raceline_filename = track_name + "_racingline.json";
            RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", raceline_filename.c_str(), track_dir.c_str());
            racelinecloudPCL = loadTrackFile(node,(fs::path(track_dir) / fs::path(raceline_filename)).string());
            RCLCPP_INFO(node->get_logger(), "Got %d points for the optimal raceline.", racelinecloudPCL.size());

            pcl::PCLPointCloud2 innercloudPC2, outercloudPC2, racelinecloudPC2;
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZ>(innercloudPCL), innercloudPC2);
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZ>(outercloudPCL), outercloudPC2);
            pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZ>(racelinecloudPCL), racelinecloudPC2);
            pcl_conversions::moveFromPCL(innercloudPC2, innercloudMSG);
            pcl_conversions::moveFromPCL(outercloudPC2, outercloudMSG);
            pcl_conversions::moveFromPCL(racelinecloudPC2, racelinecloudMSG);
            innercloudMSG.header.frame_id = "track"; 
            outercloudMSG.header.frame_id = "track"; 
            racelinecloudMSG.header.frame_id = "track"; 

        }
        innercloudMSG.header.stamp = node->now();
        outercloudMSG.header.stamp = innercloudMSG.header.stamp;
        racelinecloudMSG.header.stamp = innercloudMSG.header.stamp;

        innerpub->publish(innercloudMSG);
        outerpub->publish(outercloudMSG);
        racelinepub->publish(racelinecloudMSG);
    }


    


}