set(_required_cmake 3.15)
cmake_minimum_required (VERSION ${_required_cmake} FATAL_ERROR)
cmake_policy(VERSION ${_required_cmake})
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
project(f1_datalogger)

set(cxx_standard 17)
set(CMAKE_CXX_STANDARD ${cxx_standard})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
if(WIN32)
    option(STATIC_CRT "Build against static CRT libraries (Only relevant on Windows)" OFF)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
	include(cmake/windows_stuff.cmake)
	get_WIN32_WINNT(win32_ver)
    message(STATUS "Win32 version: ${win32_ver}")
    if(BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    if(STATIC_CRT)
        foreach(flag_var
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)

            if(${flag_var} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            endif(${flag_var} MATCHES "/MD")
        endforeach(flag_var)
    endif()
else()
    set_property(GLOBAL PROPERTY USE_FOLDERS OFF)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "..." FORCE)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(THIRD_PARTY_REPOS ${CMAKE_SOURCE_DIR}/third_party)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})

find_package( Eigen3 REQUIRED )
include_directories(${EIGEN3_INCLUDE_DIRS})

option(BUILD_DATALOGGER "Builds the datalogger libraries and executables. If set to OFF, only builds the ability to interact with existing protobuf datasets" ON)
option(GUI_TOOLS "Attempt to build the GUI tools" OFF)
if(BUILD_DATALOGGER)
    if(MSVC)
        option(WIN10API "Build against the Windows 10 Runtime" OFF)
        if(WIN10API)
            add_definitions(-D_WIN32_WINNT=${win32_ver})
        endif()
		set(DIRECTXK_DIR ${THIRD_PARTY_REPOS}/DirectXTK)
		add_subdirectory(${DIRECTXK_DIR})
        set(DIRECTXT_INCLUDE_DIR ${DIRECTXK_DIR}/Inc)
       # add_compile_options()
    endif()
    set(SCL_SUBDIR ${THIRD_PARTY_REPOS}/screen_capture_lite)
    add_subdirectory(${SCL_SUBDIR})
    set(screen_capture_lite_LIBRARIES screen_capture_lite)
    include(screen_capture_lite_PLATFORM_LIBS)
    message(STATUS "Found screen_capture_lite platform libraries: ${screen_capture_lite_PLATFORM_LIBS}")
    include_directories(${SCL_SUBDIR}/include)
endif()


if(BUILD_SHARED_LIBS)
    if(MSVC)
	    #set(CMAKE_STATIC_LIBRARY_PREFIX "")
	    add_definitions( -DBOOST_ALL_NO_LIB )
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
		set(Boost_NO_SYSTEM_PATHS true)
	endif()
	set(Boost_USE_STATIC_LIBS OFF)
else()
    set(Boost_USE_STATIC_LIBS ON)
endif()
set (Boost_USE_MULTITHREADED ON)
set(BOOST_REQUIRED_COMPONENTS
date_time
filesystem
program_options
regex
system
thread
)
find_package( Boost 1.7 REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS} )
foreach(boost_component_ ${BOOST_REQUIRED_COMPONENTS})
	list(APPEND Boost_LIBRARIES Boost::${boost_component_})
endforeach()
message(STATUS "Using Boost components: ${Boost_LIBRARIES}")



find_package( OpenCV REQUIRED )
find_package(Qt5 COMPONENTS Widgets Core Gui)
set(_gotqt (DEFINED  Qt5Widgets_FOUND) AND (DEFINED  Qt5Widgets_FOUND) AND (DEFINED  Qt5Widgets_FOUND))


if(MSVC)
find_package(yaml-cpp REQUIRED)
else()
find_package(yaml-cpp REQUIRED)
endif()
message(STATUS "Found yaml-cpp include dir: ${YAML_CPP_INCLUDE_DIR}")
message(STATUS "Found yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")
if(MSVC)
	message(STATUS "LIB Environment Variable: $ENV{LIB}")
endif()
option(GET_TBB "Download TBB from github rather than using a pre-installed version." OFF)
if(GET_TBB)
    if(MSVC)
        message(STATUS "Getting TBB")
        include(get_tbb)
        message(STATUS "Got TBB targets: ${TBB_IMPORTED_TARGETS}")
    else()
        message(STATUS "Building TBB")
        include(build_tbb)
        message(STATUS "Built TBB targets: ${TBB_IMPORTED_TARGETS}")
    endif()
else()
    find_package(TBB REQUIRED COMPONENTS tbb)
	message(STATUS "TBB Configuration at : ${TBB_DIR}")
	message(STATUS "Got TBB targets: ${TBB_IMPORTED_TARGETS}")
	get_filename_component(TBB_ROOT_DIR ${TBB_DIR} DIRECTORY)
	message(STATUS "Got TBB root: ${TBB_ROOT_DIR}")
endif()


include_directories( include )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR} )

include(FindProtobuf)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
add_subdirectory( proto/${PROJECT_NAME}/proto )
#include_directories(${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto)
set_target_properties(${PROJECT_NAME}_proto
                    PROPERTIES FOLDER proto)

add_subdirectory( src/alglib )
message(STATUS "ALGLIB_DIR: ${ALGLIB_DIR}")
set_target_properties(${PROJECT_NAME}_alglib PROPERTIES FOLDER utils)

if(BUILD_DATALOGGER)
	option(WITH_ARMA "Use Armadillo library for matrix loading/storing" ON)
	if(WITH_ARMA)
		include(include_arma)
	endif()
    add_subdirectory( src/udp_logging )
    add_subdirectory( src/image_logging )
    add_subdirectory( src/controllers )
	set_target_properties(${PROJECT_NAME}_udp_utils ${PROJECT_NAME}_eigen_utils ${PROJECT_NAME}_simulated_telemetry_server ${PROJECT_NAME}_image_utils
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
    ${PROJECT_NAME}_eigen_utils
    ${PROJECT_NAME}_image_utils
    ${PROJECT_NAME}_controllers
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


if(${_gotqt} AND GUI_TOOLS)
    add_subdirectory(src/guitools)
endif()

if(MSVC)
	configure_file(scripts/windows/tbbvars2019.bat.in tbbvars2019.bat @ONLY NEWLINE_STYLE WIN32)
	configure_file(scripts/windows/addpythonpath.bat.in addpythonpath.bat @ONLY NEWLINE_STYLE WIN32)
	configure_file(scripts/windows/addpythonpath.ps.in addpythonpath.ps1 @ONLY NEWLINE_STYLE WIN32)
else()
	configure_file(scripts/bash/addpythonpath.in addpythonpath @ONLY NEWLINE_STYLE UNIX)
endif() 