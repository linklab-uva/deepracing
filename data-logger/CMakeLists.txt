project(f1_datalogger)
cmake_minimum_required(VERSION 2.7)


add_executable(logger src/logger.cpp)
add_executable(simple_udp_server src/simple_udp_server.cpp)
add_executable(simple_udp_client src/simple_udp_client.cpp)
find_package(ZeroMQ REQUIRED)
find_package(cppzmq REQUIRED)

if(cppzmq_FOUND)
    include_directories(${cppzmq_INCLUDE_DIR})
    target_link_libraries(logger libzmq)
endif()