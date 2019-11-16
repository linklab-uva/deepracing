#include "rclcpp/rclcpp.hpp"
#include <sstream>
#include <Eigen/Geometry>
#include <tf2/buffer_core.h>
#include <functional>
#include <f1_datalogger_msgs/msg/timestamped_packet_motion_data.hpp>
#include <f1_datalogger_msgs/msg/path_raw.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include "tf2_ros/static_transform_broadcaster.h"
#include <tf2/LinearMath/Quaternion.h>
class NodeWrapperPathConverter_ 
{

  public:
    NodeWrapperPathConverter_( const rclcpp::NodeOptions & options = (
      rclcpp::NodeOptions()
      .allow_undeclared_parameters(true)
      .automatically_declare_parameters_from_overrides(true)
      ))
     {
     this->node = rclcpp::Node::make_shared("f1_path_converter");//,"",options);
     this->listener = this->node->create_subscription<f1_datalogger_msgs::msg::PathRaw>("/predicted_path_raw", std::bind(&NodeWrapperPathConverter_::pathCallback, this, std::placeholders::_1));
     this->publisher = this->node->create_publisher<nav_msgs::msg::Path>("/predicted_path");
     
     
     
    }  
    rclcpp::Subscription<f1_datalogger_msgs::msg::PathRaw>::SharedPtr listener;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher;
    std::shared_ptr<rclcpp::Node> node;
  private:
    void pathCallback(const f1_datalogger_msgs::msg::PathRaw::SharedPtr raw_path)
    {
     nav_msgs::msg::Path topub;
     topub.header=raw_path->header;

     for(unsigned int i=0; i < raw_path->posx.size(); ++i)
     {
       double currentx = raw_path->posx.at(i);
       double currentz = raw_path->posz.at(i);
       double currentvelx = raw_path->velx.at(i);
       double currentvelz = raw_path->velz.at(i);
       Eigen::Vector3d down(0.0,-1.0,0.0);
       Eigen::Vector3d forward(currentvelx,0.0,currentvelz);
       forward.normalize();
       Eigen::Vector3d left = forward.cross(down);
       left.normalize();
       Eigen::Vector3d up = forward.cross(left);
       up.normalize();
       Eigen::Matrix3d rotmat;
       rotmat.col(0)=left;
       rotmat.col(1)=up;
       rotmat.col(2)=forward;
       Eigen::Quaterniond rot(rotmat);
       geometry_msgs::msg::PoseStamped currentpose;
       currentpose.header = topub.header;
       currentpose.pose.position.x = currentx;
       currentpose.pose.position.y = 0.0;
       currentpose.pose.position.z = currentz;
       currentpose.pose.orientation.x = rot.x();
       currentpose.pose.orientation.y = rot.y();
       currentpose.pose.orientation.z = rot.z();
       currentpose.pose.orientation.w = rot.w();
       topub.poses.push_back(currentpose);
     }

     this->publisher->publish(topub);


    }
};
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  NodeWrapperPathConverter_ nw;
  std::shared_ptr<rclcpp::Node> node = nw.node;
  RCLCPP_INFO(node->get_logger(), "Converting Paths to ROS Format");
  rclcpp::spin(node);
  return 0;
}