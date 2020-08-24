cmake_minimum_required(VERSION 3.15)
project(f1_datalogger_ros)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(f1_datalogger_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(laser_geometry REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL CONFIG REQUIRED QUIET COMPONENTS core features filters io segmentation surface)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

#Provides imported targets for f1_datalogger, armadillo, Boost, Eigen3, Opencv, and (on Windows) vJoy_plusplus
find_package(f1_datalogger CONFIG REQUIRED)

find_package(cv_bridge REQUIRED)

add_library(f1_msg_utils SHARED src/utils/f1_msg_utils.cpp)
ament_target_dependencies(f1_msg_utils cv_bridge sensor_msgs std_msgs f1_datalogger_msgs rclcpp rosidl_default_runtime)
target_link_libraries(f1_msg_utils
 f1_datalogger::f1_datalogger_udp_utils
 f1_datalogger::f1_datalogger_proto
 Eigen3::Eigen
)
if(WIN32)
  target_compile_definitions(f1_msg_utils 
    PRIVATE "F1_DATALOGGER_ROS_PUBLIC=__declspec(dllexport)"
    INTERFACE "F1_DATALOGGER_ROS_PUBLIC=__declspec(dllimport)"
  )
endif()
target_include_directories(f1_msg_utils
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(tf_updater src/tf_updater.cpp)
ament_target_dependencies(tf_updater f1_datalogger_msgs geometry_msgs rclcpp tf2 tf2_ros )
target_link_libraries(tf_updater
 f1_msg_utils
 Eigen3::Eigen
)

add_executable(path_converter src/path_converter.cpp)
ament_target_dependencies(path_converter rclcpp geometry_msgs nav_msgs f1_datalogger_msgs)
target_link_libraries(path_converter
 Eigen3::Eigen
 f1_msg_utils
)

add_executable(ros_rebroadcaster src/ros_rebroadcaster.cpp)
ament_target_dependencies(ros_rebroadcaster cv_bridge image_transport sensor_msgs std_msgs f1_datalogger_msgs)
target_link_libraries(ros_rebroadcaster
 f1_msg_utils
 f1_datalogger::f1_datalogger
)

add_executable(extract_bezier_curves src/extract_bezier_curves.cpp)
ament_target_dependencies(extract_bezier_curves rosbag2_cpp rosbag2_storage sensor_msgs std_msgs f1_datalogger_msgs)
target_link_libraries(extract_bezier_curves
 f1_msg_utils
)
#target_include_directories(ros_rebroadcaster PRIVATE ${cv_bridge_INCLUDE_DIRS})

add_executable(boundary_publisher src/boundary_publisher.cpp)
ament_target_dependencies(boundary_publisher rclcpp pcl_conversions pcl_msgs pcl_ros sensor_msgs std_msgs)
target_include_directories(boundary_publisher
 PUBLIC
 ${PCL_COMMON_INCLUDE_DIRS}
)
target_link_libraries(boundary_publisher
#  ${PCL_COMMON_LIBRARIES}
#  ${PCL_KDTREE_LIBRARIES}
#  ${PCL_SEARCH_LIBRARIES}
  ${PCL_LIBRARIES}
  jsoncpp_lib
)

 ament_export_dependencies(ament_cmake)
 ament_export_dependencies(rclcpp)
 ament_export_dependencies(image_tools)
 ament_export_dependencies(laser_geometry)
 ament_export_dependencies(f1_datalogger_msgs)
 ament_export_dependencies(f1_datalogger)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
install(TARGETS
    ros_rebroadcaster
    tf_updater
    path_converter
    extract_bezier_curves
    boundary_publisher
    DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS
    f1_msg_utils
    EXPORT ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(
  DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/
)
ament_export_dependencies(f1_datalogger)
ament_export_targets(${PROJECT_NAME})
ament_package()
