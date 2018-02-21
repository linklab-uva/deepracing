project(f1_datalogger)
cmake_minimum_required(VERSION 2.7)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
find_package( Boost COMPONENTS thread chrono system filesystem REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
include_directories(include)
link_directories(${Boost_LIBRARY_DIR_RELEASE})
add_definitions(-std=c++11)
add_library(${PROJECT_NAME}_data_capture
src/simple_udp_listener
src/screen_video_capture.cpp
src/simple_screen_listener.cpp
)
target_link_libraries(${PROJECT_NAME}_data_capture
${OpenCV_LIBS}
${Boost_CHRONO_LIBRARY_RELEASE}
ws2_32
)

add_library(${PROJECT_NAME}_gsoap_binding
src/stdsoap2.cpp
src/deepf1_gsoap_env.cpp 
src/deepf1_gsoapC.cpp
)

add_library(${PROJECT_NAME}_gsoap_conversions
src/gsoap_conversions.cpp
)
target_link_libraries(${PROJECT_NAME}_gsoap_conversions
${PROJECT_NAME}_gsoap_binding
)

IF(BUILD_GSOAP_TEST)
add_executable(${PROJECT_NAME}_gsoap_test
src/gsoap_test.cpp 
)
target_link_libraries(${PROJECT_NAME}_gsoap_test 
${PROJECT_NAME}_gsoap_binding
)
ENDIF()
add_executable(${PROJECT_NAME} src/logger.cpp)
target_link_libraries(${PROJECT_NAME}
${PROJECT_NAME}_data_capture
${PROJECT_NAME}_gsoap_conversions
${Boost_CHRONO_LIBRARY_RELEASE}
${Boost_SYSTEM_LIBRARY_RELEASE}
)

add_executable(${PROJECT_NAME}_simulated_telmetry_server src/simulated_telemetry_server.cpp)
target_link_libraries(${PROJECT_NAME}_simulated_telmetry_server
${PROJECT_NAME}_data_capture
${OpenCV_LIBS}
${Boost_CHRONO_LIBRARY_RELEASE}
${Boost_SYSTEM_LIBRARY_RELEASE}
)