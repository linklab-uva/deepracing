project(f1_datalogger)
cmake_minimum_required(VERSION 2.7)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
find_package( Boost COMPONENTS thread chrono REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
include_directories(include)
link_directories(${Boost_LIBRARY_DIR_RELEASE})

add_library(${PROJECT_NAME}_data_capture src/simple_udp_listener src/screen_video_capture.cpp src/simple_screen_listener.cpp)
target_link_libraries(${PROJECT_NAME}_data_capture ${OpenCV_LIBS} ${Boost_THREAD_LIBRARY_RELEASE} ${Boost_CHRONO_LIBRARY_RELEASE})

add_executable(${PROJECT_NAME} src/logger.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_data_capture ${OpenCV_LIBS} ${Boost_THREAD_LIBRARY_RELEASE} ${Boost_CHRONO_LIBRARY_RELEASE})

