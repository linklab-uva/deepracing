project(f1_datalogger)
cmake_minimum_required(VERSION 2.7)

include_directories(include)
add_executable(logger src/logger.cpp)
add_executable(simple_udp_listener src/simple_udp_listener.cpp)
