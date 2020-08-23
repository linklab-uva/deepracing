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
#include "f1_datalogger_msgs/msg/timestamped_packet_car_status_data.hpp"
#include "f1_datalogger_msgs/msg/timestamped_packet_car_telemetry_data.hpp"
#include "f1_datalogger_msgs/msg/timestamped_packet_lap_data.hpp"
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
    rcl_interfaces::msg::ParameterDescriptor all_cars_description;
    all_cars_description.name = "publish_all_cars";
    all_cars_description.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
    all_cars_description.description = "Whether to publish information about all 20 cars in the session, or just the ego vehicle. True means publish information about all cars";
    all_cars_description.read_only = false;
    all_cars_param_ = node->declare_parameter<bool>("publish_all_cars", true, all_cars_description);
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
    rclcpp::Time now = this->node_->now();
    f1_datalogger_msgs::msg::TimestampedPacketCarStatusData rosdata;
    rosdata.header.stamp = now;
    rosdata.udp_packet = f1_datalogger_ros::F1MsgUtils::toROS(data.data, all_cars_param_);
    rosdata.timestamp = std::chrono::duration<double>(data.timestamp - begin_).count();
    status_publisher_->publish(rosdata);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketCarTelemetryData& data) override
  {
    rclcpp::Time now = this->node_->now();
    f1_datalogger_msgs::msg::TimestampedPacketCarTelemetryData rosdata;
    rosdata.header.stamp = now;
    rosdata.udp_packet = f1_datalogger_ros::F1MsgUtils::toROS(data.data, all_cars_param_);
    rosdata.timestamp = std::chrono::duration<double>(data.timestamp - begin_).count();
    telemetry_publisher_->publish(rosdata);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketEventData& data) override
  {
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketLapData& data) override
  {
    rclcpp::Time now = this->node_->now();
    f1_datalogger_msgs::msg::TimestampedPacketLapData rosdata;
    rosdata.header.stamp = now;
    rosdata.udp_packet = f1_datalogger_ros::F1MsgUtils::toROS(data.data, all_cars_param_);
    rosdata.timestamp = std::chrono::duration<double>(data.timestamp - begin_).count();
    lap_data_publisher_->publish(rosdata);
  }
  virtual inline void handleData(const deepf1::twenty_eighteen::TimestampedPacketMotionData& data) override
  {
    rclcpp::Time now = this->node_->now();
    f1_datalogger_msgs::msg::TimestampedPacketMotionData rosdata;
    rosdata.header.stamp = now;
    rosdata.udp_packet = f1_datalogger_ros::F1MsgUtils::toROS(data.data, all_cars_param_);
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

    this->lap_data_publisher_ = node_->create_publisher<f1_datalogger_msgs::msg::TimestampedPacketLapData>("lap_data", 1);
    this->motion_publisher_ = node_->create_publisher<f1_datalogger_msgs::msg::TimestampedPacketMotionData>("motion_data", 1);
    this->session_publisher_ = node_->create_publisher<f1_datalogger_msgs::msg::TimestampedPacketSessionData>("session_data", 1);
    this->status_publisher_ = node_->create_publisher<f1_datalogger_msgs::msg::TimestampedPacketCarStatusData>("status_data", 1);
    this->telemetry_publisher_ = node_->create_publisher<f1_datalogger_msgs::msg::TimestampedPacketCarTelemetryData>("telemetry_data", 1);
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
  bool all_cars_param_;
public:
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedPacketLapData> > lap_data_publisher_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedPacketMotionData> > motion_publisher_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedPacketSessionData> > session_publisher_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedPacketCarStatusData> > status_publisher_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedPacketCarTelemetryData> > telemetry_publisher_;
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
    rclcpp::ParameterValue resize_height_p = node->declare_parameter("resize_height",rclcpp::ParameterValue(66));
    resize_height_ =  resize_height_p.get<int>();

    rclcpp::ParameterValue resize_width_p = node->declare_parameter("resize_width",rclcpp::ParameterValue(200));
    resize_width_ =  resize_width_p.get<int>();

    rclcpp::ParameterValue top_left_row_p = node->declare_parameter("top_left_row",rclcpp::ParameterValue(32));
    top_left_row_ =  top_left_row_p.get<int>();

    rclcpp::ParameterValue top_left_col_p = node->declare_parameter("top_left_col",rclcpp::ParameterValue(0));
    top_left_col_ =  top_left_col_p.get<int>();

    rclcpp::ParameterValue crop_height_p = node->declare_parameter("crop_height",rclcpp::ParameterValue(0));
    crop_height_ =  crop_height_p.get<int>();

    rclcpp::ParameterValue crop_width_p = node->declare_parameter("crop_width",rclcpp::ParameterValue(0));
    crop_width_ =  crop_width_p.get<int>();

    this->node_ = node;
    //this->publisher_ = this->node_->create_publisher<sensor_msgs::msg::Image>("f1_screencaps", qos_settings);
  //  this->timestamped_publisher_ = this->node_->create_publisher<f1_datalogger_msgs::msg::TimestampedImage>("timestamped_f1_screencaps", qos_settings);
    
    this->it_cropped_publisher_ = it.advertise("/f1_screencaps/cropped", 1, true);

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
    const rclcpp::Time stamp(this->node_->now());
    std_msgs::msg::Header header;
    header.stamp=stamp;
    header.frame_id="car";
    const cv::Mat& imin = data.image;

    
    cv::Mat bgraimage, rgbimage;

    cv::Range rowrange, colrange;
    if(crop_height_ >0 && crop_width_ >0)
    {
      rowrange = cv::Range(top_left_row_,top_left_row_+crop_height_ - 1);
      colrange = cv::Range(top_left_col_,top_left_col_+crop_width_ - 1);
    }
    else
    {
      rowrange = cv::Range(top_left_row_, imin.rows-1);
      colrange = cv::Range(top_left_col_, imin.cols-1);
    }
    if(resize_width_ >0 && resize_height_ >0)
    {
      cv::resize(imin(rowrange,colrange),bgraimage,cv::Size(resize_width_,resize_height_),0.0,0.0,cv::INTER_AREA);
    }
    else
    {
      bgraimage = imin(rowrange,colrange);
    }


    cv::cvtColor(bgraimage,rgbimage,cv::COLOR_BGRA2RGB);
    cv_bridge::CvImage bridge_image(header, "rgb8", rgbimage);
    this->it_cropped_publisher_.publish( bridge_image.toImageMsg() );
  }
  void init(const deepf1::TimePoint& begin, const cv::Size& window_size) override
  {
    this->begin_ = begin;
    ready = true;
  }
  int resize_width_;
  int resize_height_;
  int crop_height_;
  int crop_width_;
  int top_left_row_;
  int top_left_col_;
private:
  bool ready;
  std::shared_ptr<rclcpp::Node> node_;
  image_transport::ImageTransport it;
  image_transport::Publisher it_cropped_publisher_;
  std::shared_ptr<rclcpp::Publisher <sensor_msgs::msg::Image> > publisher_;
  std::shared_ptr<rclcpp::Publisher <f1_datalogger_msgs::msg::TimestampedImage> > timestamped_publisher_;
  deepf1::TimePoint begin_;
  
};
class NodeWrapper_ 
{

  public:
    NodeWrapper_()
     {
      datagrab_node = rclcpp::Node::make_shared("f1_data_publisher","");
      imagegrab_node = rclcpp::Node::make_shared("f1_image_publisher","");

      datagrab_handler.reset(new ROSRebroadcaster_2018DataGrabHandler(datagrab_node));
      image_handler.reset(new ROSRebroadcaster_FrameGrabHandler(imagegrab_node));
    }  
    std::shared_ptr<rclcpp::Node> datagrab_node, imagegrab_node;
    std::shared_ptr<ROSRebroadcaster_2018DataGrabHandler> datagrab_handler;
    std::shared_ptr<ROSRebroadcaster_FrameGrabHandler> image_handler;
};
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  NodeWrapper_ nw;
  std::shared_ptr<rclcpp::Node> datagrab_node = nw.datagrab_node;
  std::shared_ptr<rclcpp::Node> imagegrab_node = nw.imagegrab_node;
  rcl_interfaces::msg::ParameterDescriptor capture_freq_description;
  capture_freq_description.floating_point_range.push_back(rcl_interfaces::msg::FloatingPointRange());
  capture_freq_description.floating_point_range[0].from_value=1.0;
  capture_freq_description.floating_point_range[0].to_value=60.0;
  capture_freq_description.floating_point_range[0].step=0.0;
  capture_freq_description.description="Frequency (in Hz) to capture images from the screen";

  rclcpp::ParameterValue search_string_p = imagegrab_node->declare_parameter("search_string",rclcpp::ParameterValue("F1"));
  rclcpp::ParameterValue capture_frequency_p = imagegrab_node->declare_parameter("capture_frequency",rclcpp::ParameterValue(35.0), capture_freq_description);
  
  rclcpp::ParameterValue hostname_p = datagrab_node->declare_parameter("hostname",rclcpp::ParameterValue("127.0.0.1"));
  rcl_interfaces::msg::ParameterDescriptor port_description;
  port_description.integer_range.push_back(rcl_interfaces::msg::IntegerRange());
  port_description.integer_range[0].from_value=20777;
  port_description.integer_range[0].to_value=21000;
  port_description.integer_range[0].step=1;

  rclcpp::ParameterValue port_p = datagrab_node->declare_parameter("port",rclcpp::ParameterValue(20777), port_description);
  rclcpp::ParameterValue rebroadcast_p = datagrab_node->declare_parameter("rebroadcast",rclcpp::ParameterValue(false));

  deepf1::F1DataLogger dl(search_string_p.get<std::string>(), hostname_p.get<std::string>(),
                          port_p.get<int>(), rebroadcast_p.get<bool>());  
  
  if(rebroadcast_p.get<bool>())
  {
    printf("Rebroadcasting on port: %ld\n", port_p.get<int>()+1);
  }
  RCLCPP_INFO(imagegrab_node->get_logger(),
              "Listening for data from the game. \n"
              "Cropping an area (HxW)  (%u, %u)\n"
              "Resizing cropped area to (HxW)  (%u, %u)\n"
              ,nw.image_handler->crop_height_, nw.image_handler->crop_width_, nw.image_handler->resize_height_, nw.image_handler->resize_width_);
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::executor::ExecutorArgs(),4);
  exec.add_node(datagrab_node);
  exec.add_node(imagegrab_node);
  dl.start(capture_frequency_p.get<double>(), nw.datagrab_handler, nw.image_handler);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}