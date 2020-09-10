#ifndef F1_DATALOGGER_ROS_F1_MSG_UTILS_H
#define F1_DATALOGGER_ROS_F1_MSG_UTILS_H
#include "f1_datalogger/car_data/f1_2018/car_data.h"
#include "f1_datalogger/car_data/f1_2020/car_data.h"
#include "f1_datalogger_msgs/msg/packet_header.hpp"
#include "f1_datalogger_msgs/msg/packet_motion_data.hpp"
#include "f1_datalogger_msgs/msg/packet_car_setup_data.hpp"
#include "f1_datalogger_msgs/msg/packet_car_status_data.hpp"
#include "f1_datalogger_msgs/msg/packet_car_telemetry_data.hpp"
#include "f1_datalogger_msgs/msg/packet_lap_data.hpp"
#include "f1_datalogger_msgs/msg/packet_session_data.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <array>

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
        
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::CarSetupData toROS(const deepf1::twenty_eighteen::CarSetupData& setup_data);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketCarSetupData toROS(const deepf1::twenty_eighteen::PacketCarSetupData& packet_setup_data, bool copy_all_cars);

        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::LapData toROS(const deepf1::twenty_eighteen::LapData& lap_data);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketLapData toROS(const deepf1::twenty_eighteen::PacketLapData& lap_data, bool copy_all_cars);

        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::CarTelemetryData toROS(const deepf1::twenty_eighteen::CarTelemetryData& telemetry_data);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketCarTelemetryData toROS(const deepf1::twenty_eighteen::PacketCarTelemetryData& telemetry_data, bool copy_all_cars);

        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::CarStatusData toROS(const deepf1::twenty_eighteen::CarStatusData& status_data);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketCarStatusData toROS(const deepf1::twenty_eighteen::PacketCarStatusData& packet_status_data, bool copy_all_cars);

        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::CarMotionData toROSMotionData(const deepf1::twenty_eighteen::CarMotionData& motion_data);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketMotionData toROS(const deepf1::twenty_eighteen::PacketMotionData& motion_data, bool copy_all_cars);

        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::MarshalZone toROS(const deepf1::twenty_eighteen::MarshalZone& marshal_zone);
        static F1_DATALOGGER_ROS_PUBLIC f1_datalogger_msgs::msg::PacketSessionData toROS(const deepf1::twenty_eighteen::PacketSessionData& session_data);
        static inline F1_DATALOGGER_ROS_PUBLIC const std::array<std::string, 4> wheel_order()
        {
            return std::array<std::string, 4>({"RearLeft, RearRight, FrontLeft, FrontRight"});
        }
        static inline F1_DATALOGGER_ROS_PUBLIC const std::array<std::string, 25> track_names()
        {
            return std::array<std::string, 25>({"Australia", "France", "China", "Bahrain", "Spain", "Monaco",
                                              "Canada", "Britain", "Germany", "Hungary", "Belgium", "Italy",
                                               "Singapore", "Japan", "Abu_Dhabi", "USA", "Brazil", "Austria",
                                               "Russia", "Mexico", "Azerbaijan", "Bahrain_short", "Britan_short",
                                               "USA_short", "Japan_short"});
        }
        static F1_DATALOGGER_ROS_PUBLIC constexpr char* world_coordinate_name = "track";
    private:

    };
}
#endif