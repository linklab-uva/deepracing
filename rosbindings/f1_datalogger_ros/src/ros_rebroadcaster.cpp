#include "rclcpp/rclcpp.hpp"
#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"
#include "f1_datalogger/f1_datalogger.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include "f1_datalogger/udp_logging/utils/eigen_utils.h"
#include "f1_datalogger/image_logging/utils/opencv_utils.h"
#include "f1_datalogger_ros/utils/f1_msg_utils.h"
#include "f1_datalogger_msgs/msg/timestamped_image.hpp"
#include "f1_datalogger_msgs/msg/timestamped_packet_motion_data.hpp"
#include "f1_datalogger_msgs/msg/timestamped_packet_car_telemetry_data.hpp"
#include "f1_datalogger_msgs/msg/timestamped_packet_session_data.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
// #include <image_transport/camera_publisher.h>
// #include <image_transport/transport_hints.h>
class ROSRebroadcaster_2018DataGrabHandler : public deepf1::IF12018DataGrabHandler
{
public:
  ROSRebroadcaster_2018DataGrabHandler(std::shared_ptr<rclcpp::Node> node)
  {
    this->node_ = node;
  }
  bool isReady() override
  {
    return ready_;
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarSetupData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarStatusData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data) override
  {
    f1_datalogger_msgs::msg::TimestampedPacketCarTelemetryData rosdata;
    rosdata.udp_packet = f1_datalogger_ros::F1MsgUtils::toROS(data.data);
    rosdata.timestamp = std::chrono::duration<double>(data.timestamp - begin_).count();
    telemetry_publisher_->publish(rosdata);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data) override
  {
    rclcpp::Time now = this->node_->now();
    f1_datalogger_msgs::msg::TimestampedPacketMotionData rosdata;
    rosdata.udp_packet = f1_datalogger_ros::F1MsgUtils::toROS(data.data);
    for (f1_datalogger_msgs::msg::CarMotionData & motion_data : rosdata.udp_packet.car_motion_data)
    {
      motion_data.world_forward_dir.header.stamp = now;
      motion_data.world_position.header.stamp = now;
      motion_data.world_right_dir.header.stamp = now;
      motion_data.world_up_dir.header.stamp = now;
      motion_data.world_velocity.header.stamp = now;
    }
    rosdata.timestamp = std::chrono::duration<double>(data.timestamp - begin_).count();
    motion_publisher_->publish(rosdata);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketParticipantsData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketSessionData& data) override
  {
    f1_datalogger_msgs::msg::TimestampedPacketSessionData rosdata;
    rosdata.udp_packet = f1_datalogger_ros::F1MsgUtils::toROS(data.data);
    rosdata.timestamp = std::chrono::duration<double>(data.timestamp - begin_).count();
    session_publisher_->publish(rosdata);
  }
  void init(const std::string& host, unsigned int port, const deepf1::TimePoint& begin) override
  {

    this->motion_publisher_ = node_->create_publisher<f1_datalogger_msgs::msg::TimestampedPacketMotionData>("motion_data", 10);
    this->telemetry_publisher_ = node_->create_publisher<f1_datalogger_msgs::msg::TimestampedPacketCarTelemetryData>("telemetry_data", 10);
    this->session_publisher_ = node_->create_publisher<f1_datalogger_msgs::msg::TimestampedPacketSessionData>("session_data", 10);
    ready_ = true;
    this->begin_ = begin;
    this->host_ = host;
    this->port_ = port;
  }
private:
  bool ready_;
  std::chrono::high_resolution_clock::time_point begin_;
  std::string host_;
  unsigned int port_;
public:
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedPacketMotionData> > motion_publisher_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedPacketCarTelemetryData> > telemetry_publisher_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedPacketSessionData> > session_publisher_;
};
class ROSRebroadcaster_FrameGrabHandler : public deepf1::IF1FrameGrabHandler
{
public:
  ROSRebroadcaster_FrameGrabHandler(std::shared_ptr<rclcpp::Node> node) : it(node)
  {
    std::cout<<"Loadable transports:"<<std::endl;
    for(const std::string& s : it.getLoadableTransports())
    {
      std::cout<<s<<std::endl;
    }
    rclcpp::QoS qos_settings(100);
    this->node_ = node;
    //this->publisher_ = this->node_->create_publisher<sensor_msgs::msg::Image>("f1_screencaps", qos_settings);
  //  this->timestamped_publisher_ = this->node_->create_publisher<f1_datalogger_msgs::msg::TimestampedImage>("timestamped_f1_screencaps", qos_settings);
    
    this->compressed_publisher_ = it.advertise("/f1_screencaps", 1, true);

  }
  virtual ~ROSRebroadcaster_FrameGrabHandler()
  {
  }
  bool isReady() override
  {
    return ready;
  }
  void handleData(const deepf1::TimestampedImageData& data) override
  {
    const rclcpp::Time stamp=this->node_->now();
    const cv::Mat& imin = data.image;
    cv::Range rowrange(32,32+crop_height_);
    cv::Range colrange(0,crop_width_-1);
    cv::Mat & imcrop = imin(rowrange,colrange);
    cv::Mat rgbimage, bgraimage;
    cv::resize(imcrop,bgraimage,cv::Size(resize_width_,resize_height_),0.0,0.0,cv::INTER_AREA);
    cv::cvtColor(bgraimage,rgbimage,cv::COLOR_BGRA2RGB);
    std_msgs::msg::Header header;
    header.stamp=stamp;
    header.frame_id="car";
    cv_bridge::CvImage bridge_image(header, "rgb8", rgbimage);
    const sensor_msgs::msg::Image::SharedPtr & image_msg = bridge_image.toImageMsg();
    
    // f1_datalogger_msgs::msg::TimestampedImage timestamped_image;
    // timestamped_image.timestamp = std::chrono::duration<double>(data.timestamp - begin_).count();
    // timestamped_image.image = *image_msg;
    // this->timestamped_publisher_->publish(timestamped_image);
   // this->publisher_->publish(image_msg);
    this->compressed_publisher_.publish(image_msg);
  }
  void init(const deepf1::TimePoint& begin, const cv::Size& window_size) override
  {
    this->begin_ = begin;
    ready = true;
  }
  static constexpr double captureFreq = 35.0;
  unsigned int resize_width_;
  unsigned int resize_height_;
  unsigned int crop_height_;
  unsigned int crop_width_;
private:
  bool ready;
  std::shared_ptr<rclcpp::Node> node_;
  image_transport::ImageTransport it;
  image_transport::Publisher compressed_publisher_;
  std::shared_ptr<rclcpp::Publisher <sensor_msgs::msg::Image> > publisher_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedImage> > timestamped_publisher_;
  deepf1::TimePoint begin_;
  
};
class NodeWrapper_ 
{

  public:
    NodeWrapper_( const rclcpp::NodeOptions & options = (
      rclcpp::NodeOptions()
      .allow_undeclared_parameters(true)
      .automatically_declare_parameters_from_overrides(true)
      ))
     {
     this->node = rclcpp::Node::make_shared("f1_data_publisher","",options);
      datagrab_handler.reset(new ROSRebroadcaster_2018DataGrabHandler(node));
      image_handler.reset(new ROSRebroadcaster_FrameGrabHandler(node));
    }  
    std::shared_ptr<rclcpp::Node> node;
    std::shared_ptr<ROSRebroadcaster_2018DataGrabHandler> datagrab_handler;
    std::shared_ptr<ROSRebroadcaster_FrameGrabHandler> image_handler;
};
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  NodeWrapper_ nw;
  std::shared_ptr<rclcpp::Node> node = nw.node;
  double resize_factor;
  std::string search_string;
  double capture_frequency;
  unsigned int resize_height, resize_width, crop_height, crop_width;
  node->get_parameter_or<std::string>("search_string",search_string, std::string("F1") );
  node->get_parameter_or<double>("capture_frequency",capture_frequency, ROSRebroadcaster_FrameGrabHandler::captureFreq);
  node->get_parameter_or<unsigned int>("resize_height",nw.image_handler->resize_height_, 66);
  node->get_parameter_or<unsigned int>("resize_width",nw.image_handler->resize_width_, 200);
  node->get_parameter_or<unsigned int>("crop_height",nw.image_handler->crop_height_, 362);
  node->get_parameter_or<unsigned int>("crop_width",nw.image_handler->crop_width_ , 1758);
  deepf1::F1DataLogger dl(search_string);  
  dl.start(capture_frequency, nw.datagrab_handler, nw.image_handler);
  
  RCLCPP_INFO(node->get_logger(),
              "Listening for data from the game. Resizing images to (HxW)  (%u, %u)",nw.image_handler->resize_height_, nw.image_handler->resize_width_);

  rclcpp::spin(node);
 // rclcpp::shutdown();
  return 0;
}