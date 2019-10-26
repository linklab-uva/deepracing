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
find_package(rosidl_default_runtime REQUIRED)
find_package(image_tools REQUIRED)
find_package(laser_geometry REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
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
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
if(WIN32)
  find_package(vJoy_plusplus REQUIRED)
endif()
find_package(Armadillo CONFIG REQUIRED)

include_directories(include)
add_library(f1_msg_utils src/utils/f1_msg_utils.cpp)
ament_target_dependencies(f1_msg_utils f1_datalogger sensor_msgs std_msgs f1_datalogger_msgs rclcpp rosidl_default_runtime)
target_link_libraries(f1_msg_utils
 f1_datalogger::f1_datalogger_udp_utils
 Eigen3::Eigen
 Boost::system
 Boost::regex
)
target_include_directories(f1_msg_utils PRIVATE ${ARMADILLO_INCLUDE_DIRS})

add_executable(ros_rebroadcaster src/ros_rebroadcaster.cpp)
ament_target_dependencies(ros_rebroadcaster f1_datalogger)
ament_target_dependencies(ros_rebroadcaster sensor_msgs)
ament_target_dependencies(ros_rebroadcaster f1_datalogger_msgs)
ament_target_dependencies(ros_rebroadcaster rclcpp)
target_link_libraries(ros_rebroadcaster
 f1_msg_utils
 f1_datalogger::f1_datalogger
 f1_datalogger::f1_datalogger_udp_utils
 f1_datalogger::f1_datalogger_image_utils
 f1_datalogger::f1_datalogger_eigen_utils
 Eigen3::Eigen
 armadillo
)
#target_include_directories(ros_rebroadcaster PRIVATE ${ARMADILLO_INCLUDE_DIRS})


 ament_export_dependencies(ament_cmake)
 ament_export_dependencies(rcl)
 ament_export_dependencies(image_tools)
 ament_export_dependencies(laser_geometry)
 ament_export_dependencies(f1_datalogger_msgs)
 ament_export_dependencies(f1_datalogger_provider)
 #ament_export_dependencies(f1_datalogger)
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
    f1_msg_utils
    DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/
)
get_filename_component(f1_cmakefile_ "${f1_datalogger_DIR}/f1_datalogger-config.cmake" REALPATH)
get_filename_component(f1_cmake_directory_ "${f1_cmakefile_}" DIRECTORY)
message(STATUS "f1_cmake_directory_: ${f1_cmake_directory_}")
set(f1_python_directory_ "${f1_cmake_directory_}/../Lib/python")
message(STATUS "f1_python_directory_: ${f1_python_directory_}")
get_filename_component(f1_python_directory_ "${f1_python_directory_}" REALPATH)
message(STATUS "f1_python_directory_ after resolving symlinks: ${f1_python_directory_}")
if(WIN32)
  configure_file(addpythonpath.bat.in addpythonpath.bat @ONLY)
  ament_environment_hooks(${CMAKE_BINARY_DIR}/addpythonpath.bat)
elseif(UNIX)
  configure_file(addpythonpath.bash.in addpythonpath.bash @ONLY)
  ament_environment_hooks(${CMAKE_BINARY_DIR}/addpythonpath.bash)
endif()
ament_package()
