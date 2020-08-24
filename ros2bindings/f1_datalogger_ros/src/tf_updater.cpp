#include "rclcpp/rclcpp.hpp"
#include <sstream>
#include <Eigen/Geometry>
#include <tf2/buffer_core.h>
#include <functional>
#include <f1_datalogger_msgs/msg/timestamped_packet_motion_data.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include "tf2_ros/static_transform_broadcaster.h"
#include <tf2/LinearMath/Quaternion.h>
#include <boost/math/constants/constants.hpp>
#include <tf2/convert.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

class NodeWrapperTfUpdater_ 
{

  public:
    NodeWrapperTfUpdater_( const rclcpp::NodeOptions & options = (
      rclcpp::NodeOptions()
      .allow_undeclared_parameters(true)
      .automatically_declare_parameters_from_overrides(true)
      ))
     {
     this->node = rclcpp::Node::make_shared("f1_tf_updater");//,"",options);
     this->statictfbroadcaster.reset(new tf2_ros::StaticTransformBroadcaster(node));
     this->tfbroadcaster.reset(new tf2_ros::TransformBroadcaster(node));
     worldToTrack.header.frame_id = "world";
     worldToTrack.header.stamp = this->node->now();
     worldToTrack.child_frame_id = "track";
     tf2::Quaternion quat;
     quat.setRPY( boost::math::constants::half_pi<double>(), 0.0, 0.0 );
     worldToTrack.transform.translation.x = 0.0;
     worldToTrack.transform.translation.y = 0.0;
     worldToTrack.transform.translation.z = 0.0;
     worldToTrack.transform.rotation.x = quat.x();
     worldToTrack.transform.rotation.y = quat.y();
     worldToTrack.transform.rotation.z = quat.z();
     worldToTrack.transform.rotation.w = quat.w();
     this->statictfbroadcaster->sendTransform(worldToTrack);
     this->listener = this->node->create_subscription<f1_datalogger_msgs::msg::TimestampedPacketMotionData>("/motion_data", 1, std::bind(&NodeWrapperTfUpdater_::packetCallback, this, std::placeholders::_1));
     this->pose_publisher = this->node->create_publisher<geometry_msgs::msg::PoseStamped>("/car_pose", 1);
     
     
    }  
    rclcpp::Subscription<f1_datalogger_msgs::msg::TimestampedPacketMotionData>::SharedPtr listener;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher;
    std::shared_ptr<rclcpp::Node> node;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfbroadcaster;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> statictfbroadcaster;
    geometry_msgs::msg::TransformStamped worldToTrack;    
  private:
    void packetCallback(const f1_datalogger_msgs::msg::TimestampedPacketMotionData::SharedPtr motion_data_packet)
    {
     // std::cout << "Got some data" << std::endl;
     // RCLCPP_INFO(node->get_logger(), "Got some data");
      uint8_t idx;
      if( motion_data_packet->udp_packet.header.player_car_index<20 )
      {
        idx = motion_data_packet->udp_packet.header.player_car_index;
      }
      else
      {
        idx = 0;
      }

      const f1_datalogger_msgs::msg::CarMotionData &motion_data = motion_data_packet->udp_packet.car_motion_data[idx];
      const geometry_msgs::msg::Vector3Stamped &velocityROS = motion_data.world_velocity;
      const geometry_msgs::msg::Vector3Stamped &upROS = motion_data.world_up_dir;
      const geometry_msgs::msg::Vector3Stamped &forwardROS = motion_data.world_forward_dir;
      const geometry_msgs::msg::Vector3Stamped &rightROS = motion_data.world_right_dir;

      Eigen::Vector3d leftEigen(-rightROS.vector.x, -rightROS.vector.y, -rightROS.vector.z);
      Eigen::Vector3d forwardEigen(forwardROS.vector.x, forwardROS.vector.y, forwardROS.vector.z);
      Eigen::Vector3d upEigen(upROS.vector.x, upROS.vector.y, upROS.vector.z);
      Eigen::Matrix3d rotmat;
      rotmat.col(0) = leftEigen;
      rotmat.col(1) = upEigen;
      rotmat.col(2) = forwardEigen;
      Eigen::Quaterniond rotationEigen(rotmat);
      rotationEigen.normalize();
      geometry_msgs::msg::TransformStamped transformMsg;
      transformMsg.header.frame_id = "track";
      transformMsg.header.stamp = motion_data.world_position.header.stamp;
     // transformMsg.header.stamp = this->node->now();
      transformMsg.child_frame_id = "car";
      transformMsg.transform.rotation.x = rotationEigen.x();
      transformMsg.transform.rotation.y = rotationEigen.y();
      transformMsg.transform.rotation.z = rotationEigen.z();
      transformMsg.transform.rotation.w = rotationEigen.w();

      const geometry_msgs::msg::PointStamped& positionROS = motion_data.world_position;
      transformMsg.transform.translation.x = positionROS.point.x;
      transformMsg.transform.translation.y = positionROS.point.y;
      transformMsg.transform.translation.z = positionROS.point.z;
      this->tfbroadcaster->sendTransform(transformMsg);
      this->statictfbroadcaster->sendTransform(worldToTrack);

      geometry_msgs::msg::PoseStamped pose;
      pose.set__header(transformMsg.header);
      pose.pose.position.set__x(transformMsg.transform.translation.x);
      pose.pose.position.set__y(transformMsg.transform.translation.y);
      pose.pose.position.set__z(transformMsg.transform.translation.z);
      pose.pose.set__orientation(transformMsg.transform.rotation);

      this->pose_publisher->publish(pose);
      


    }
};
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  NodeWrapperTfUpdater_ nw;
  std::shared_ptr<rclcpp::Node> node = nw.node;
  RCLCPP_INFO(node->get_logger(), "Updating TF data");
  rclcpp::spin(node);
  return 0;
}