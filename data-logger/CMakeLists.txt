project(f1_datalogger)
cmake_minimum_required (VERSION 3.10 FATAL_ERROR)

if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
    set(cxx_standard 11)
else()
    set_property(GLOBAL PROPERTY USE_FOLDERS OFF)
    option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
    set(cxx_standard 14)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "..." FORCE)
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
    set(SCL_SUBDIR ${THIRD_PARTY_REPOS}/screen_capture_lite)
    add_subdirectory(${SCL_SUBDIR})
    set(screen_capture_lite_LIBRARIES screen_capture_lite)
    include(screen_capture_lite_PLATFORM_LIBS)
    message(STATUS "Found screen_capture_lite platform libraries: ${screen_capture_lite_PLATFORM_LIBS}")
    include_directories(${SCL_SUBDIR}/include)
endif()


if(BUILD_SHARED_LIBS)
    if(MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    set(Boost_USE_STATIC_LIBS OFF)
else()
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package( Boost REQUIRED COMPONENTS date_time filesystem program_options regex thread)
message(STATUS "Found Boost libraries: ${Boost_LIBRARIES} in directory: ${Boost_LIBRARY_DIRS}")



find_package( OpenCV REQUIRED )
find_package(Qt5Widgets CONFIG)
find_package(Qt5Core CONFIG)
find_package(Qt5Gui CONFIG)
set(_gotqt ${Qt5Widgets_FOUND} AND ${Qt5Core_FOUND} AND ${Qt5Gui_FOUND})


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

add_subdirectory( proto/${PROJECT_NAME}/proto )
set_target_properties(${PROJECT_NAME}_proto
                    PROPERTIES FOLDER proto)

include_directories(${CMAKE_BINARY_DIR}/proto)
include_directories(${Protobuf_INCLUDE_DIR})


include_directories(${ProtobufIncludePath})

add_subdirectory( src/alglib )
message(STATUS "ALGLIB_DIR: ${ALGLIB_DIR}")
add_subdirectory( src/post_processing )
set_target_properties(${PROJECT_NAME}_alglib ${PROJECT_NAME}_post_processing ${PROJECT_NAME}_assign_labels
                    PROPERTIES FOLDER utils)

if(BUILD_DATALOGGER)
    add_subdirectory( src/image_logging )
    add_subdirectory( src/udp_logging )
	set_target_properties(${PROJECT_NAME}_udp_utils ${PROJECT_NAME}_simulated_telemetry_server ${PROJECT_NAME}_image_utils
						PROPERTIES FOLDER utils)
	set_target_properties(screen_capture_example
						PROPERTIES FOLDER utils)
    add_library(${PROJECT_NAME}
    src/f1_datalogger.cpp
    )
    target_link_libraries(${PROJECT_NAME} 
    ${PROJECT_NAME}_image_logging 
    ${PROJECT_NAME}_udp_logging 
    ${PROJECT_NAME}_proto 
    ${PROJECT_NAME}_alglib 
	${PROJECT_NAME}_image_utils
    ${Boost_LIBRARIES}
    )
    add_subdirectory( src/Examples )

	
	set_target_properties(${PROJECT_NAME} ${PROJECT_NAME}_udp_logging ${PROJECT_NAME}_image_logging  screen_capture_lite
						PROPERTIES FOLDER core)
	set_target_properties(${PROJECT_NAME}_udp_logging_common ${PROJECT_NAME}_image_logging_common
						PROPERTIES FOLDER common)

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
endif()


# Find the QtWidgets library
if(${_gotqt})
    add_subdirectory(src/guitools)
endif()


install(DIRECTORY proto/
DESTINATION share/${PROJECT_NAME}/proto
FILES_MATCHING PATTERN "*.proto"
)
