project(f1_datalogger)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(screen_capture_lite REQUIRED)
find_package( Boost COMPONENTS filesystem program_options system REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} )
include_directories( include )

add_subdirectory( proto )
add_subdirectory( src/alglib )
add_subdirectory( src/image_logging )
add_subdirectory( src/udp_logging )

add_library(${PROJECT_NAME}
src/f1_datalogger.cpp
)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_image_logging ${PROJECT_NAME}_udp_logging ${Boost_LIBRARIES})


add_executable(${PROJECT_NAME}_multithreaded_capture
src/multi_threaded_capture.cpp
)
target_link_libraries(${PROJECT_NAME}_multithreaded_capture
${PROJECT_NAME}
${PROJECT_NAME}_image_logging 
${PROJECT_NAME}_udp_logging 
${screen_capture_lite_LIBRARIES}
${screen_capture_lite_PLATFORM_LIBS}
)

add_subdirectory( src/Examples )
