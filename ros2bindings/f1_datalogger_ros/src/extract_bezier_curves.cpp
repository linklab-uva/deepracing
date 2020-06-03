#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_cpp/readers/sequential_reader.hpp"
#include "rosbag2_cpp/typesupport_helpers.hpp"
#include "rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp"

#include "rosbag2_storage/bag_metadata.hpp"
#include "rosbag2_storage/metadata_io.hpp"
#include "rosbag2_storage/topic_metadata.hpp"
#include "rosbag2_storage/storage_factory.hpp"
#include <rclcpp/rclcpp.hpp>
#include <f1_datalogger_msgs/msg/bezier_curve.h>
#include <filesystem>
#include <unordered_map>
#include <sstream>
#include <f1_datalogger_msgs/msg/bezier_curve.hpp>
#include <f1_datalogger_msgs/msg/timestamped_packet_motion_data.hpp>
#ifndef F1_DATALOGGER_PROTO_DLL_MACRO
    #define F1_DATALOGGER_PROTO_DLL_MACRO __declspec(dllimport)
#endif
#include <f1_datalogger/proto/BezierCurve.pb.h>
#include <google/protobuf/util/json_util.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <Eigen/Dense>
using rosbag2_cpp::converter_interfaces::SerializationFormatConverter;
rosbag2_cpp::ConverterTypeSupport createTypeSupport(const std::string& type_name)
{
    rosbag2_cpp::ConverterTypeSupport type_support;
    type_support.type_support_library = rosbag2_cpp::get_typesupport_library(type_name, "rosidl_typesupport_cpp");
    type_support.rmw_type_support = rosbag2_cpp::get_typesupport_handle(type_name, "rosidl_typesupport_cpp", type_support.type_support_library);
    return type_support;
}
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    namespace fs = std::filesystem;
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("extract_bezier_curves","");
    std::cout<<"Hello World!"<<std::endl;
    RCLCPP_INFO(node->get_logger(), "Hello ROS2!");
    rcl_interfaces::msg::ParameterDescriptor storage_uri_description;
    storage_uri_description.description="Filepath of the bagfile to load";
    storage_uri_description.name="storage_uri";
    std::string storage_uri = node->declare_parameter<std::string>(storage_uri_description.name,"",storage_uri_description);
    
    rcl_interfaces::msg::ParameterDescriptor output_directory_description;
    output_directory_description.description="Where to put the resulting bezier curve proto files";
    output_directory_description.name="output_directory";
    std::string output_directory_str = node->declare_parameter<std::string>(output_directory_description.name,"",output_directory_description);
    
    rcl_interfaces::msg::ParameterDescriptor format_description;
    format_description.description="Storage format in the bag file";
    format_description.name="format";
    std::string format = node->declare_parameter<std::string>(format_description.name,"cdr",format_description);

    rcl_interfaces::msg::ParameterDescriptor storage_id_description;
    storage_id_description.description="Which plugin to use. The default is usually sqlite3";
    storage_id_description.name="storage_id";
    std::string storage_id = node->declare_parameter<std::string>(storage_id_description.name,"sqlite3",storage_id_description);

    if(storage_uri.compare("")==0)
    {
        RCLCPP_ERROR(node->get_logger(), "Must specify a bag file with the storage_uri parameter. Exiting.");
        rclcpp::shutdown();
        exit(0);
    }
    if(output_directory_str.compare("")==0)
    {
        RCLCPP_ERROR(node->get_logger(), "Must specify where to put the proto files with the output_directory parameter. Exiting.");
        rclcpp::shutdown();
        exit(0);
    }
    fs::path output_directory(output_directory_str);
    if(!fs::is_directory(output_directory))
    {
        fs::create_directories(output_directory);
    }
    RCLCPP_INFO(node->get_logger(), "Attempting to open bagfile: %s", storage_uri.c_str());
    // rosbag2_storage::TopicMetadata topic_with_type;
    // topic_with_type.name = "/predicted_path";
    // topic_with_type.type = "f1_datalogger_msgs/BezierCurve";
    // topic_with_type.serialization_format = "cdr";
    // auto topics_and_types = std::vector<rosbag2_storage::TopicMetadata>{topic_with_type};

    // auto message = std::make_shared<rosbag2_storage::SerializedBagMessage>();
    // message->topic_name = topic_with_type.name;

    rosbag2_cpp::StorageOptions storage_options;
    storage_options.uri=storage_uri;
    storage_options.storage_id=storage_id;
    rosbag2_cpp::ConverterOptions converter_options;
    converter_options.input_serialization_format=format;
    converter_options.output_serialization_format=converter_options.input_serialization_format;

    rosbag2_cpp::readers::SequentialReader reader;
    reader.open(storage_options,converter_options);

    std::string topic_path = "/predicted_path";
    std::string topic_motion_packet = "/motion_data";

    std::unordered_map<std::string, rosbag2_cpp::ConverterTypeSupport> topics_and_types_;
    for(const rosbag2_storage::TopicMetadata & metadata : reader.get_all_topics_and_types())
    {
        topics_and_types_.insert({metadata.name,createTypeSupport(metadata.type)});
    }

    rosbag2_cpp::SerializationFormatConverterFactory factory;
    std::unique_ptr<rosbag2_cpp::converter_interfaces::SerializationFormatDeserializer> cdr_deserializer_;
    cdr_deserializer_ = factory.load_deserializer(format);

    std::vector< std::shared_ptr<f1_datalogger_msgs::msg::BezierCurve> > bezier_curves;
    std::vector< std::shared_ptr<f1_datalogger_msgs::msg::TimestampedPacketMotionData> > motion_packets;
    std::vector<double> motion_data_session_timestamps;
    std::vector<double> motion_data_ros_timestamps;
    while(reader.has_next())
    {
        std::shared_ptr<rosbag2_storage::SerializedBagMessage> serialized_msg = reader.read_next();
        RCLCPP_DEBUG(node->get_logger(), "Read a message on topic %s", serialized_msg->topic_name.c_str());
        auto ts = topics_and_types_.at(serialized_msg->topic_name).rmw_type_support;
        auto introspection_ts = topics_and_types_.at(serialized_msg->topic_name).introspection_type_support;
        std::shared_ptr<rosbag2_cpp::rosbag2_introspection_message_t> allocated_ros_message = std::make_shared<rosbag2_cpp::rosbag2_introspection_message_t>();
        allocated_ros_message->time_stamp = serialized_msg->time_stamp;
        allocated_ros_message->allocator = rcutils_get_default_allocator();
        if(serialized_msg->topic_name.compare(topic_path)==0)
        {
            std::shared_ptr<f1_datalogger_msgs::msg::BezierCurve> bezier_curve_msg(new f1_datalogger_msgs::msg::BezierCurve);
            allocated_ros_message->message = bezier_curve_msg.get();
            cdr_deserializer_->deserialize(serialized_msg, ts, allocated_ros_message);
            rclcpp::Time timestamp = bezier_curve_msg->header.stamp;
            RCLCPP_DEBUG(node->get_logger(), "Deserialized a Bezier Curve. Timestamp: %f", timestamp.seconds());
            bezier_curves.push_back(bezier_curve_msg);
        }
        else
        {
            std::shared_ptr<f1_datalogger_msgs::msg::TimestampedPacketMotionData> motion_packet_msg(new f1_datalogger_msgs::msg::TimestampedPacketMotionData);
            allocated_ros_message->message = motion_packet_msg.get();
            cdr_deserializer_->deserialize(serialized_msg, ts, allocated_ros_message);
            geometry_msgs::msg::Point point = motion_packet_msg->udp_packet.car_motion_data[0].world_position.point;
            rclcpp::Time timestamp = motion_packet_msg->header.stamp;
            RCLCPP_DEBUG(node->get_logger(), "Deserialized a Motion Packet. Timestamp: %f", timestamp.seconds());
            motion_packets.push_back(motion_packet_msg);
            motion_data_ros_timestamps.push_back(timestamp.seconds());
            motion_data_session_timestamps.push_back(double(motion_packet_msg->udp_packet.header.session_time));
        }
    }
    Eigen::Map<Eigen::VectorXd> ros_timestamps_eigen(motion_data_ros_timestamps.data(), motion_data_ros_timestamps.size());
    double t0 = ros_timestamps_eigen[0];
    ros_timestamps_eigen = ros_timestamps_eigen - t0*Eigen::VectorXd::Ones(motion_data_ros_timestamps.size());
    Eigen::Map<Eigen::VectorXd> session_timestamps_eigen(motion_data_session_timestamps.data(), motion_data_session_timestamps.size());
    Eigen::MatrixXd A(ros_timestamps_eigen.size(),2);
    A.col(0) = ros_timestamps_eigen;
    A.col(1) = Eigen::VectorXd::Ones(ros_timestamps_eigen.size());
    std::stringstream ss;
    ss << std::endl << A << std::endl;
    RCLCPP_DEBUG(node->get_logger(), "%s", ss.str().c_str());
    RCLCPP_INFO(node->get_logger(), "Lsq matrix: is [%d x %d]", A.rows(),A.cols());
    Eigen::VectorXd solution = A.colPivHouseholderQr().solve(session_timestamps_eigen);
    RCLCPP_INFO(node->get_logger(), "Solution vector: [%f , %f]", solution[0], solution[1]);
    google::protobuf::util::JsonOptions json_options;
    json_options.add_whitespace = true;
    json_options.always_print_primitive_fields = true;
    for(unsigned int i = 0; i < bezier_curves.size(); i++)
    {
        std::shared_ptr<f1_datalogger_msgs::msg::BezierCurve> bezier_curve_msg = bezier_curves[i];
        rclcpp::Time timestamp = bezier_curve_msg->header.stamp;
        double session_timestamp = solution[0]*(timestamp.seconds() - t0) + solution[1];
        RCLCPP_DEBUG(node->get_logger(), "Session timestamp for Bezier Curve %d: %f", i, session_timestamp);
        deepf1::twenty_eighteen::protobuf::BezierCurve bezier_curve_proto;
        for(unsigned int j = 0; j < bezier_curve_msg->control_points_x.size();++j)
        {
            bezier_curve_proto.add_control_points_x(bezier_curve_msg->control_points_x.at(j));
            bezier_curve_proto.add_control_points_z(bezier_curve_msg->control_points_z.at(j));
        }
        bezier_curve_proto.set_m_sessiontime(session_timestamp);
        fs::path filepath = output_directory / fs::path("bezier_curve_" + std::to_string(i+1) + ".json");
        std::string bezier_curve_pb_json;
        google::protobuf::util::Status rc = google::protobuf::util::MessageToJsonString(bezier_curve_proto, &bezier_curve_pb_json, json_options);
        std::ofstream ostream(filepath.string(), std::fstream::out | std::fstream::trunc);
        ostream << bezier_curve_pb_json << std::endl;
        ostream.flush();
        ostream.close();
    }
    RCLCPP_INFO(node->get_logger(), "Range of session times: [%f , %f]", session_timestamps_eigen[0], session_timestamps_eigen[session_timestamps_eigen.size()-1]);

    
    

}