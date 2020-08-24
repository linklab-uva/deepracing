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
    
    std::array<int,3> sizes = {xarray.size(), yarray.size(), zarray.size()};
    if (! std::all_of(sizes.begin(), sizes.end(), [xarray, yarray, zarray](int i){return i==xarray.size();}) )
    {
       RCLCPP_FATAL(node->get_logger(), "All three arrays are not the same size"); 
    }
    return rootval;
}
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    namespace fs = std::filesystem;
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("f1_boundary_publisher","");
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > innerpub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/inner_track_boundary",1);
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > outerpub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/outer_track_boundary",1);
    std::shared_ptr< rclcpp::Publisher<sensor_msgs::msg::PointCloud2> > racelinepub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/optimal_raceline",1);

    rclcpp::ParameterValue track_dir_param = node->declare_parameter("track_dir",rclcpp::ParameterValue(""));
    std::string track_dir = track_dir_param.get<std::string>();
    rclcpp::ParameterValue track_name_param = node->declare_parameter("track_name",rclcpp::ParameterValue(""));
    std::string track_name = track_name_param.get<std::string>();



    RCLCPP_INFO(node->get_logger(), "Openning file %s in directory %s.", (track_name + "_innerlimit.json").c_str(), track_dir.c_str());

    std::string inner_jsonpath = (fs::path(track_dir) / fs::path(track_name + "_innerlimit.json")).string();
    Json::Value inner_boundary_dict = readJsonFile(node, inner_jsonpath);
    Json::Value inner_boundary_x = inner_boundary_dict["x"];
    Json::Value inner_boundary_y = inner_boundary_dict["y"];
    Json::Value inner_boundary_z = inner_boundary_dict["z"];
    pcl::PointCloud<pcl::PointXYZ> innercloudPCL;
    for (unsigned int i =0; i < inner_boundary_x.size(); i++)
    {
        innercloudPCL.push_back(pcl::PointXYZ(inner_boundary_x[i].asDouble(), inner_boundary_y[i].asDouble(), inner_boundary_z[i].asDouble()));
    }
    RCLCPP_INFO(node->get_logger(), "Got %d points for the inner boundary.", innercloudPCL.size());
    innercloudPCL.header.frame_id="track";

    std::string outer_jsonpath = (fs::path(track_dir) / fs::path(track_name + "_outerlimit.json")).string();
    Json::Value outer_boundary_dict = readJsonFile(node, outer_jsonpath);
    Json::Value outer_boundary_x = outer_boundary_dict["x"];
    Json::Value outer_boundary_y = outer_boundary_dict["y"];
    Json::Value outer_boundary_z = outer_boundary_dict["z"];
    pcl::PointCloud<pcl::PointXYZ> outercloudPCL;
    for (unsigned int i =0; i < outer_boundary_x.size(); i++)
    {
        outercloudPCL.push_back(pcl::PointXYZ(outer_boundary_x[i].asDouble(), outer_boundary_y[i].asDouble(), outer_boundary_z[i].asDouble()));
    }
    RCLCPP_INFO(node->get_logger(), "Got %d points for the outer boundary.", outercloudPCL.size());
    outercloudPCL.header.frame_id="track";
    while (rclcpp::is_initialized())
    {
        sensor_msgs::msg::PointCloud2 innercloudMSG, outercloudMSG;
        pcl::PCLPointCloud2 innercloudPC2, outercloudPC2;
        pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZ>(innercloudPCL), innercloudPC2);
        pcl::toPCLPointCloud2(pcl::PointCloud<pcl::PointXYZ>(outercloudPCL), outercloudPC2);
        pcl_conversions::moveFromPCL(innercloudPC2, innercloudMSG);
        pcl_conversions::moveFromPCL(outercloudPC2, outercloudMSG);

        innerpub->publish(innercloudMSG);
        outerpub->publish(outercloudMSG);
        rclcpp::sleep_for(std::chrono::nanoseconds(1000000000));
        rclcpp::spin_some(node);
        
        RCLCPP_DEBUG(node->get_logger(), "Ran a spin.");
    }


    


}