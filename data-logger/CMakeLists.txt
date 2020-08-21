set(_required_cmake 3.15)
cmake_minimum_required (VERSION ${_required_cmake} FATAL_ERROR)
cmake_policy(VERSION ${_required_cmake})
# if(COMMAND cmake_policy)
# 	cmake_policy(SET CMP0003 NEW)
# endif(COMMAND cmake_policy)
project(f1_datalogger)
find_package(ament_cmake QUIET)
message(STATUS "Using CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif(NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_DEBUG_POSTFIX _debug)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
if(WIN32)
    option(WIN10API "Build against the Windows 10 Runtime" ON)
	include(cmake/windows_stuff.cmake)
	get_WIN32_WINNT(win32_ver)
    message(STATUS "Win32 version: ${win32_ver}")
    if(WIN10API)
      add_definitions(-D_WIN32_WINNT=${win32_ver})
      set(BUILD_XAUDIO_WIN10 ON CACHE BOOL "Build for XAudio 2.9" FORCE)
    endif()
    if(BUILD_SHARED_LIBS)
      add_definitions(-DF1_DATALOGGER_BUILDING_DLL)
    endif()
    add_compile_options(-w)
endif(WIN32)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "..." FORCE)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(THIRD_PARTY_REPOS ${CMAKE_SOURCE_DIR}/third_party)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})




set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf REQUIRED)

add_subdirectory( proto/${PROJECT_NAME}/proto )
set(PROTO_INCLUDE_DIR ${CMAKE_BINARY_DIR}/proto)

option(BUILD_DATALOGGER "Builds the datalogger libraries and executables. If set to OFF, only builds the ability to interact with existing protobuf datasets" ON)
if(BUILD_DATALOGGER)
    option(YAML_CPP_DLL "Set this if YAML CPP was compiled as a DLL" ON)
    option(GUI_TOOLS "Attempt to build the GUI tools" OFF)
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
    set( Boost_USE_STATIC_LIBS (NOT ${BUILD_SHARED_LIBS}) )
    find_package(Boost CONFIG COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
    if(Boost_FOUND)
        message(STATUS "Found boost via cmake config at ${Boost_DIR}")
    else()
        find_package(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
    endif(Boost_FOUND)
    set(Boost_TARGETS)
    foreach(comp IN LISTS BOOST_REQUIRED_COMPONENTS)
    list(APPEND Boost_TARGETS Boost::${comp})
    endforeach(comp IN LISTS BOOST_REQUIRED_COMPONENTS)
    list(APPEND Boost_TARGETS Boost::headers)

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
    find_package( OpenCV REQUIRED )
    find_package( Eigen3 REQUIRED )
    find_package(Qt5 QUIET COMPONENTS Widgets Core Gui)
    set(_gotqt (DEFINED  Qt5Widgets_FOUND) AND (DEFINED  Qt5_FOUND))
    include_directories( include )
    include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR} )
    find_package(TBB REQUIRED COMPONENTS tbb)
    include(include_yaml_cpp)
    #find_package(yaml-cpp CONFIG REQUIRED)
    message(STATUS "Found yaml-cpp include dir: ${YAML_CPP_INCLUDE_DIR}")
    message(STATUS "Found yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")

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
    ${PROJECT_NAME}_eigen_utils
    ${PROJECT_NAME}_image_utils
    ${PROJECT_NAME}_controllers
    ${Boost_TARGETS}
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

install(DIRECTORY include/${PROJECT_NAME}
DESTINATION include
FILES_MATCHING PATTERN "*.h"
)
if(ament_cmake_FOUND)
    message(STATUS "Ament was found. Building a ROS2-style cmake package")
    ament_export_targets(${PROJECT_NAME}-export)
    add_subdirectory(python)
    ament_package(CONFIG_EXTRAS cmake/find_deps.cmake.in)
else()
    install(EXPORT ${PROJECT_NAME}-export
    FILE
    ${PROJECT_NAME}Targets.cmake
    NAMESPACE
        f1_datalogger::
    DESTINATION
        cmake
    )

    configure_file(${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)

    install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake DESTINATION cmake/)
    install(DIRECTORY python/deepracing
        DESTINATION lib/python
    )


    if(MSVC)
        file(TO_NATIVE_PATH ${CMAKE_INSTALL_PREFIX}/Lib/python python_install_directory)
        configure_file(scripts/windows/tbbvars2019.bat.in tbbvars2019.bat @ONLY NEWLINE_STYLE WIN32)
        configure_file(scripts/windows/addpythonpath.build.ps1.in ${CMAKE_CURRENT_BINARY_DIR}/addpythonpathbuild.ps1 @ONLY NEWLINE_STYLE WIN32)
        configure_file(scripts/windows/addpythonpath.install.ps1.in ${CMAKE_CURRENT_BINARY_DIR}/addpythonpath.ps1 @ONLY NEWLINE_STYLE WIN32)
        configure_file(scripts/windows/addpythonpath.install.bat.in ${CMAKE_CURRENT_BINARY_DIR}/addpythonpath.bat @ONLY NEWLINE_STYLE WIN32)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/addpythonpath.ps1 DESTINATION share/${PROJECT_NAME})
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/addpythonpath.bat DESTINATION share/${PROJECT_NAME})
    else()
        configure_file(scripts/bash/addpythonpath.build.bash.in ${CMAKE_BINARY_DIR}/addpythonpathbuild.bash @ONLY NEWLINE_STYLE UNIX)
        configure_file(scripts/bash/addpythonpath.install.bash.in ${CMAKE_BINARY_DIR}/addpythonpath.bash @ONLY NEWLINE_STYLE UNIX)
        install(FILES ${CMAKE_BINARY_DIR}/addpythonpath.bash DESTINATION share/${PROJECT_NAME})

        add_custom_target(
        chmodaddpythonpath ALL
        DEPENDS ${CMAKE_BINARY_DIR}/addpythonpath.bash
        COMMAND dos2unix ${CMAKE_BINARY_DIR}/addpythonpath.bash && chmod a+x ${CMAKE_BINARY_DIR}/addpythonpath.bash
        VERBATIM
        )
    endif() 

endif()
