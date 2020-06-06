#ifndef F1_DATALOGGER_ROS_F1_MSG_UTILS_H
#define F1_DATALOGGER_ROS_F1_MSG_UTILS_H
#include "f1_datalogger/car_data/timestamped_car_data.h"
#include "f1_datalogger_msgs/msg/packet_header.hpp"
#include "f1_datalogger_msgs/msg/packet_motion_data.hpp"
#include "f1_datalogger_msgs/msg/packet_car_telemetry_data.hpp"
#include "f1_datalogger_msgs/msg/packet_session_data.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <opencv2/core.hpp>

#ifndef F1_DATALOGGER_ROS_PUBLIC
    #define F1_DATALOGGER_ROS_PUBLIC
#endif
namespace f1_datalogger_ros
{
    class F1MsgUtils
    {
    public:
        F1_DATALOGGER_ROS_PUBLIC F1MsgUtils() = default;
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketHeader toROS(const deepf1::twenty_eighteen::PacketHeader& header_data);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::CarTelemetryData toROS(const deepf1::twenty_eighteen::CarTelemetryData& telemetry_data);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketCarTelemetryData toROS(const deepf1::twenty_eighteen::PacketCarTelemetryData& telemetry_data, bool copy_all_cars);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketMotionData toROS(const deepf1::twenty_eighteen::PacketMotionData& motion_data, bool copy_all_cars);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::CarMotionData toROS(const deepf1::twenty_eighteen::CarMotionData& motion_data);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::MarshalZone toROS(const deepf1::twenty_eighteen::MarshalZone& marshal_zone);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketSessionData toROS(const deepf1::twenty_eighteen::PacketSessionData& session_data);
        static F1_DATALOGGER_ROS_PUBLIC int depthStrToInt(const std::string& depth);
        static F1_DATALOGGER_ROS_PUBLIC int getCvType(const std::string & encoding);
        static F1_DATALOGGER_ROS_PUBLIC sensor_msgs::msg::Image toImageMsg(const cv::Mat & image, std_msgs::msg::Header header = std_msgs::msg::Header());
    private:
        static constexpr char* world_coordinate_name = "track";

    };
}
#endif