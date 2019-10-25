#include "rclcpp/rclcpp.hpp"
#include "f1_datalogger/udp_logging/utils/udp_stream_utils.h"
#include <Eigen/Geometry>
#include "f1_datalogger/f1_datalogger.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <sensor_msgs/msg/image.hpp>
#include <Eigen/Geometry>
#include "f1_datalogger/udp_logging/utils/eigen_utils.h"
#include "f1_datalogger/image_logging/utils/opencv_utils.h"
#include "f1_datalogger_ros/utils/f1_msg_utils.h"
#include "f1_datalogger_msgs/msg/timestamped_packet_motion_data.hpp"
#include "f1_datalogger_msgs/msg/timestamped_packet_car_telemetry_data.hpp"
#include "f1_datalogger_msgs/msg/timestamped_packet_session_data.hpp"
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
    f1_datalogger_msgs::msg::TimestampedPacketMotionData rosdata;
    rosdata.udp_packet = f1_datalogger_ros::F1MsgUtils::toROS(data.data);
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
  ROSRebroadcaster_FrameGrabHandler(std::shared_ptr<rclcpp::Node> node)
  {
    this->node_ = node;
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
    const cv::Mat& imin = data.image;
    cv::Mat rgbimage(imin.rows, imin.cols, CV_8UC3);
    imin.convertTo(rgbimage, rgbimage.type());
    cv::resize(rgbimage,rgbimage,cv::Size(),0.5,0.5,cv::INTER_AREA);
    sensor_msgs::msg::Image rosimage = f1_datalogger_ros::F1MsgUtils::toImageMsg(rgbimage);
    this->publisher_->publish(rosimage);
  }
  void init(const deepf1::TimePoint& begin, const cv::Size& window_size) override
  {
    this->publisher_ = node_->create_publisher<sensor_msgs::msg::Image>("f1_screencaps", 10);
    this->begin_ = begin;
    ready = true;
  }
  static constexpr double captureFreq = 35.0;
private:
  bool ready;
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::Publisher <sensor_msgs::msg::Image> > publisher_;
  deepf1::TimePoint begin_;
  
};
class NodeWrapper_
{

  public:
    NodeWrapper_()
    {
      datagrab_handler.reset(new ROSRebroadcaster_2018DataGrabHandler(node));
      image_handler.reset(new ROSRebroadcaster_FrameGrabHandler(node));
    }
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("f1_data_publisher");
    std::shared_ptr<ROSRebroadcaster_2018DataGrabHandler> datagrab_handler;
    std::shared_ptr<ROSRebroadcaster_FrameGrabHandler> image_handler;
};
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  NodeWrapper_ nw;
  std::string search(argv[1]);
  deepf1::F1DataLogger dl(search);  
  dl.start(ROSRebroadcaster_FrameGrabHandler::captureFreq, nw.datagrab_handler, nw.image_handler);
  RCLCPP_INFO(nw.node->get_logger(),
              "Help me Obi-Wan Kenobi, you're my only hope");

  rclcpp::spin(nw.node);
 // rclcpp::shutdown();
  return 0;
}