project(f1_datalogger)
cmake_minimum_required(VERSION 2.7)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
find_package( Boost COMPONENTS thread chrono system filesystem REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
include_directories(include)
link_directories(${Boost_LIBRARY_DIR_RELEASE})
link_directories(${Boost_LIBRARY_DIR_DEBUG})
add_definitions(-std=c++11)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
add_library(${PROJECT_NAME}_data_capture src/simple_udp_listener src/screen_video_capture.cpp src/simple_screen_listener.cpp)
target_link_libraries(${PROJECT_NAME}_data_capture ${OpenCV_LIBS} ${Boost_THREAD_LIBRARY_RELEASE} ${Boost_CHRONO_LIBRARY_RELEASE} ws2_32)


add_library(${PROJECT_NAME}_gsoap_binding
src/stdsoap2.cpp
src/deepf1_gsoap_env.cpp 
src/deepf1_gsoap.cpp
)
#target_link_libraries(${PROJECT_NAME}_gsoap_binding)

add_executable(${PROJECT_NAME} src/logger.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_data_capture
${OpenCV_LIBS} ${Boost_THREAD_LIBRARY_RELEASE}
${Boost_CHRONO_LIBRARY_RELEASE}
${PROJECT_NAME}_gsoap_binding)

add_executable(${PROJECT_NAME}_gsoap_test
src/gsoap_test.cpp 
)
target_link_libraries(${PROJECT_NAME}_gsoap_test ${PROJECT_NAME}_gsoap_binding)