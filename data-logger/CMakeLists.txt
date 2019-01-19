project(f1_datalogger)
cmake_minimum_required (VERSION 3.10 FATAL_ERROR)

option(BUILD_SHARED_LIBS "Build Shared Libraries (EXPERIMENTAL)" OFF)
if(MSVC)
    set(cxx_standard 11)
else()
    set(cxx_standard 17)
endif()

set(CMAKE_CXX_STANDARD ${cxx_standard})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(THIRD_PARTY_REPOS ${CMAKE_SOURCE_DIR}/third_party)
option(CMAKE_BUILD_TYPE "Build Type" Release)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_DATALOGGER "Builds the datalogger libraries and executables. If set to OFF, only builds the ability to read datasets" ON)
if(BUILD_DATALOGGER)
    find_package(screen_capture_lite REQUIRED)
    message(STATUS "Found screen_capture_lite headers: ${screen_capture_lite_INCLUDE_DIRS}")
    message(STATUS "Found screen_capture_lite libraries: ${screen_capture_lite_LIBRARIES}")
    include_directories(${screen_capture_lite_INCLUDE_DIRS})
endif()


if(BUILD_SHARED_LIBS)
    if(MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    set(Boost_USE_STATIC_LIBS OFF)
else()
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package( Boost REQUIRED COMPONENTS date_time filesystem program_options regex )
message(STATUS "Found Boost libraries: ${Boost_LIBRARIES}")



find_package( OpenCV REQUIRED )

if(MSVC)
find_package(yaml-cpp REQUIRED HINTS "C:\\Program\ Files\\YAML_CPP")
else()
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})
endif()
message(STATUS "Found yaml-cpp include dir: ${YAML_CPP_INCLUDE_DIR}")
message(STATUS "Found yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")
message(STATUS "LIB: $ENV{LIB}")
if(MSVC)
    message(STATUS "Getting TBB")
    include(get_tbb)
    message(STATUS "Got TBB targets: ${TBB_IMPORTED_TARGETS}")
else()
    option(GET_TBB "Download TBB from github rather than building it." OFF)
    if(GET_TBB)
        message(STATUS "Getting TBB")
        include(get_tbb)
        message(STATUS "Got TBB targets: ${TBB_IMPORTED_TARGETS}")
    else()
        message(STATUS "Building TBB")
        include(build_tbb)
        message(STATUS "Built TBB targets: ${TBB_IMPORTED_TARGETS}")
    endif()
endif()


include_directories( include )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR})

add_subdirectory( proto )


include_directories(${Protobuf_INCLUDE_DIR})


include_directories(${ProtobufIncludePath})

add_subdirectory( src/alglib )
message(STATUS "ALGLIB_DIR: ${ALGLIB_DIR}")
add_subdirectory( src/post_processing )

if(BUILD_DATALOGGER)
    add_subdirectory( src/image_logging )
    add_subdirectory( src/udp_logging )
endif()
add_library(${PROJECT_NAME}
src/f1_datalogger.cpp
)
target_link_libraries(${PROJECT_NAME} 
${PROJECT_NAME}_image_logging 
${PROJECT_NAME}_udp_logging 
${PROJECT_NAME}_proto 
${PROJECT_NAME}_alglib 
${Boost_LIBRARIES}
)

if(BUILD_DATALOGGER)
    add_subdirectory( src/Examples )
endif()


set(install_targets ${PROJECT_NAME} )
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/${PROJECT_NAME}
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
unset(install_targets)