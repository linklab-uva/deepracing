#ifndef F1_DATALOGGER_ROS_F1_MSG_UTILS_H
#define F1_DATALOGGER_ROS_F1_MSG_UTILS_H
#include "f1_datalogger/car_data/timestamped_car_data.h"
#include "f1_datalogger_msgs/msg/packet_header.hpp"
#include "f1_datalogger_msgs/msg/packet_motion_data.hpp"
#include "f1_datalogger_msgs/msg/packet_car_telemetry_data.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <opencv2/core.hpp>
namespace f1_datalogger_ros
{
    class F1MsgUtils
    {
    public:
        F1MsgUtils() = default;
        void doNothing();
        static f1_datalogger_msgs::msg::CarTelemetryData toROS(const deepf1::twenty_eighteen::CarTelemetryData& telemetry_data);
        static f1_datalogger_msgs::msg::PacketCarTelemetryData toROS(const deepf1::twenty_eighteen::PacketCarTelemetryData& telemetry_data);
        static f1_datalogger_msgs::msg::PacketHeader toROS(const deepf1::twenty_eighteen::PacketHeader& header_data);
        static f1_datalogger_msgs::msg::PacketMotionData toROS(const deepf1::twenty_eighteen::PacketMotionData& motion_data);
        static f1_datalogger_msgs::msg::CarMotionData toROS(const deepf1::twenty_eighteen::CarMotionData& motion_data);
        static int depthStrToInt(const std::string& depth);
        static int getCvType(const std::string & encoding);
        static sensor_msgs::msg::Image toImageMsg(const cv::Mat & image, std_msgs::msg::Header header = std_msgs::msg::Header());
    private:

    };
}
#endif