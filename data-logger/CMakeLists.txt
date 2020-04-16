set(_required_cmake 3.12)
cmake_minimum_required (VERSION ${_required_cmake} FATAL_ERROR)
cmake_policy(VERSION ${_required_cmake})
# if(COMMAND cmake_policy)
# 	cmake_policy(SET CMP0003 NEW)
# endif(COMMAND cmake_policy)
project(f1_datalogger)
message(STATUS "Using CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif(NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_DEBUG_POSTFIX _debug)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
if(WIN32)
    option(STATIC_CRT "Build against static CRT libraries (Only relevant on Windows)" OFF)
    option(WIN10API "Build against the Windows 10 Runtime" ON)
	include(cmake/windows_stuff.cmake)
	get_WIN32_WINNT(win32_ver)
    message(STATUS "Win32 version: ${win32_ver}")
    if(WIN10API)
        add_definitions(-D_WIN32_WINNT=${win32_ver})
        set(BUILD_XAUDIO_WIN10 ON CACHE BOOL "Build for XAudio 2.9" FORCE)
    endif()
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
    endif(STATIC_CRT)
endif(WIN32)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "..." FORCE)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(THIRD_PARTY_REPOS ${CMAKE_SOURCE_DIR}/third_party)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})



set(BOOST_REQUIRED_COMPONENTS
date_time
program_options
regex
system
thread
)


include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#include <filesystem>
                            int main(int argc, char** argv) { std::filesystem::path p; return 0; }" HAS_STDCXXFS)

if(NOT HAS_STDCXXFS)
    list(APPEND BOOST_REQUIRED_COMPONENTS filesystem)
endif(NOT HAS_STDCXXFS)

set(Boost_USE_STATIC_LIBS (NOT ${BUILD_SHARED_LIBS}))
find_package(Boost COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
if(Boost_FOUND)
    message(STATUS "Found boost via cmake config at ${Boost_DIR}")
else()
    find_package(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
endif(Boost_FOUND)

set(internal_boost_headers_target FALSE)
if(NOT TARGET Boost::headers)
    set(internal_boost_headers_target TRUE)
    add_library(Boost::headers INTERFACE IMPORTED GLOBAL)
    target_include_directories(Boost::headers INTERFACE ${Boost_INCLUDE_DIRS})
endif()
list(APPEND Boost_LIBRARIES Boost::headers)
message(STATUS "Using Boost components: ${Boost_LIBRARIES}")



find_package( Eigen3 REQUIRED )

option(YAML_CPP_DLL "Set this if YAML CPP was compiled as a DLL" ON)
option(BUILD_DATALOGGER "Builds the datalogger libraries and executables. If set to OFF, only builds the ability to interact with existing protobuf datasets" ON)
option(GUI_TOOLS "Attempt to build the GUI tools" OFF)
if(BUILD_DATALOGGER)
    if(MSVC)
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




find_package( OpenCV REQUIRED )
find_package(Qt5 QUIET COMPONENTS Widgets Core Gui)
set(_gotqt (DEFINED  Qt5Widgets_FOUND) AND (DEFINED  Qt5_FOUND))

include(include_yaml_cpp)
#find_package(yaml-cpp CONFIG REQUIRED)
message(STATUS "Found yaml-cpp include dir: ${YAML_CPP_INCLUDE_DIR}")
message(STATUS "Found yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")

find_package(TBB REQUIRED COMPONENTS tbb)


include_directories( include )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR} )



include(FindProtobuf)
find_package(Protobuf REQUIRED)

add_subdirectory( proto/${PROJECT_NAME}/proto )
set(PROTO_INCLUDE_DIR ${CMAKE_BINARY_DIR}/proto)

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
    PUBLIC
    ${PROJECT_NAME}_image_logging 
    ${PROJECT_NAME}_udp_logging 
    ${PROJECT_NAME}_proto 
    ${PROJECT_NAME}_alglib 
    ${PROJECT_NAME}_eigen_utils
    ${PROJECT_NAME}_image_utils
    ${PROJECT_NAME}_controllers
    ${Boost_LIBRARIES}
    INTERFACE
    ${screen_capture_lite_PLATFORM_LIBS}
    )
    add_subdirectory( src/Examples )

    set(install_targets
        ${PROJECT_NAME} 
        ${PROJECT_NAME}_image_logging 
        ${PROJECT_NAME}_image_logging_common 
        ${PROJECT_NAME}_udp_utils
        ${PROJECT_NAME}_udp_logging 
        ${PROJECT_NAME}_udp_logging_common 
        ${PROJECT_NAME}_proto 
        ${PROJECT_NAME}_alglib 
        ${PROJECT_NAME}_eigen_utils
        ${PROJECT_NAME}_image_utils
        ${PROJECT_NAME}_controllers
        screen_capture_lite
    )
else()
    set(install_targets
            ${PROJECT_NAME}_proto 
        )
endif()


install(TARGETS ${install_targets}
    EXPORT ${PROJECT_NAME}-export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(EXPORT ${PROJECT_NAME}-export
FILE
${PROJECT_NAME}Targets.cmake
NAMESPACE
    f1_datalogger::
DESTINATION
    cmake
)
install(DIRECTORY include/${PROJECT_NAME}
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

configure_file(${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake DESTINATION cmake/)
install(DIRECTORY python/deepracing
    DESTINATION lib/python
)


if(MSVC)
	configure_file(scripts/windows/tbbvars2019.bat.in tbbvars2019.bat @ONLY NEWLINE_STYLE WIN32)
	configure_file(scripts/windows/addpythonpath.bat.in addpythonpath.bat @ONLY NEWLINE_STYLE WIN32)
	configure_file(scripts/windows/addpythonpath.ps.in addpythonpath.ps1 @ONLY NEWLINE_STYLE WIN32)
else()
    configure_file(scripts/bash/addpythonpath.in ${CMAKE_BINARY_DIR}/addpythonpath @ONLY NEWLINE_STYLE UNIX)
    add_custom_target(
    chmodaddpythonpath ALL
    DEPENDS ${CMAKE_BINARY_DIR}/addpythonpath
    COMMAND dos2unix ${CMAKE_BINARY_DIR}/addpythonpath && chmod a+x ${CMAKE_BINARY_DIR}/addpythonpath
    VERBATIM
    )
endif() 
