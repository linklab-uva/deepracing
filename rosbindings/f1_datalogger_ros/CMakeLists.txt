cmake_minimum_required(VERSION 3.5)
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
find_package(rclcpp REQUIRED)
find_package(image_tools REQUIRED)
find_package(laser_geometry REQUIRED)
find_package(f1_datalogger_msgs REQUIRED)
find_package(f1_datalogger_provider REQUIRED)
find_package(f1_datalogger REQUIRED)
set(BOOST_REQUIRED_COMPONENTS
date_time
filesystem
program_options
regex
system
thread
)
find_package( Boost 1.7 REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS} )
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(ros_rebroadcaster src/ros_rebroadcaster.cpp)
ament_target_dependencies(ros_rebroadcaster f1_datalogger)
ament_target_dependencies(ros_rebroadcaster f1_datalogger_msgs)
ament_target_dependencies(ros_rebroadcaster rclcpp)
target_link_libraries(ros_rebroadcaster
 f1_datalogger::f1_datalogger_udp_utils
 f1_datalogger::f1_datalogger_image_utils
 )

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
    DESTINATION lib/${PROJECT_NAME}
)
ament_package()
