set(_required_cmake 3.15)
cmake_minimum_required (VERSION ${_required_cmake} FATAL_ERROR)
cmake_policy(VERSION ${_required_cmake})
project(f1_datalogger)
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
    #This is probably a ROS2-based installation, also look for the yaml-cpp provided by ROS2.
    find_package(yaml_cpp_vendor QUIET)
endif()
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
      add_compile_options(-D_WIN32_WINNT=${win32_ver})
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



add_library(headers INTERFACE)
target_include_directories(headers
INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
)

find_package(PCL CONFIG REQUIRED)
add_library(pcltypes src/pcl_types.cpp)
target_link_libraries(pcltypes PUBLIC headers PRIVATE ${PCL_LIBRARIES})
target_include_directories(pcltypes PUBLIC ${PCL_INCLUDE_DIRS})
set_target_properties(pcltypes PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_pcltypes)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf REQUIRED)
add_subdirectory( proto/${PROJECT_NAME}/proto )
set(PROTO_INCLUDE_DIR ${CMAKE_BINARY_DIR}/proto)
if(WIN32)
    configure_file(scripts/windows/addpythonpath.build.bat.in ${CMAKE_BINARY_DIR}/addpythonpath.bat @ONLY)
    configure_file(scripts/windows/addpythonpath.build.ps1.in ${CMAKE_BINARY_DIR}/addpythonpath.ps1 @ONLY)
    set_target_properties(pcltypes PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
else()
    configure_file(scripts/bash/addpythonpath.build.bash.in ${CMAKE_BINARY_DIR}/addpythonpath.bash @ONLY)
endif()

option(BUILD_DATALOGGER "Builds the datalogger libraries and executables. If set to OFF, only builds the ability to interact with existing protobuf datasets" OFF)
option(BUILD_CONTROLLERS "Builds the controllers library for pushing commands back into the F1 game" OFF)
if(BUILD_DATALOGGER)
    option(YAML_CPP_DLL "Set this if YAML CPP was compiled as a DLL" ON)
    option(GUI_TOOLS "Attempt to build the GUI tools" OFF)
    set(BOOST_REQUIRED_COMPONENTS
    program_options
    )
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("#include <filesystem>
                                int main(int argc, char** argv) { std::filesystem::path p; return 0; }" HAS_STDCXXFS)
    if(NOT HAS_STDCXXFS)
        list(APPEND BOOST_REQUIRED_COMPONENTS filesystem)
    endif(NOT HAS_STDCXXFS)

    find_package(Boost CONFIG QUIET COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
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
    find_package(TBB CONFIG REQUIRED COMPONENTS tbb)
    set(TBB_IMPORTED_TARGETS TBB::tbb)
    include(include_yaml_cpp)
    #find_package(yaml-cpp CONFIG REQUIRED)
    message(STATUS "Found yaml-cpp include dir: ${YAML_CPP_INCLUDE_DIR}")
    message(STATUS "Found yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")

	option(WITH_ARMA "Use Armadillo library for matrix loading/storing" OFF)
	if(WITH_ARMA)
		include(include_arma)
	endif()
    add_subdirectory( src/udp_logging )
    add_subdirectory( src/image_logging )
    add_library(${PROJECT_NAME}
    src/f1_datalogger.cpp
    )
    target_link_libraries(${PROJECT_NAME} 
    PUBLIC
    ${PROJECT_NAME}_image_logging 
    ${PROJECT_NAME}_image_logging_common 
    ${PROJECT_NAME}_udp_logging 
    ${PROJECT_NAME}_udp_logging_common
    ${PROJECT_NAME}_proto 
    ${PROJECT_NAME}_eigen_utils
    ${PROJECT_NAME}_image_utils
    ${Boost_TARGETS}
    INTERFACE
    ${screen_capture_lite_PLATFORM_LIBS}
    PRIVATE
    pcltypes
    )
    if(BUILD_CONTROLLERS)
        add_subdirectory( src/controllers )
        target_link_libraries(${PROJECT_NAME} PUBLIC
            ${PROJECT_NAME}_controllers
        )
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE F1_DATALOGGER_BUILDING_DLL)
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
        headers
        screen_capture_lite
        pcltypes
    )
    if(BUILD_CONTROLLERS)
        list(APPEND install_targets ${PROJECT_NAME}_controllers)
    endif()
    
    if(WIN32 AND GUI_TOOLS)
        list(APPEND install_targets ${PROJECT_NAME}_winrtcapture)
    endif()
    
    install(DIRECTORY include/${PROJECT_NAME}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )
else()
    set(install_targets
            ${PROJECT_NAME}_proto 
            headers
            pcltypes
        )
    install(DIRECTORY include/${PROJECT_NAME}/car_data
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h"
    )
    install(FILES 
        include/${PROJECT_NAME}/filesystem_helper.h
        include/${PROJECT_NAME}/proto_dll_macro.h
        include/${PROJECT_NAME}/visibility_control.h
    DESTINATION include/${PROJECT_NAME} 
    )
endif()


install(TARGETS ${install_targets}
    EXPORT ${PROJECT_NAME}-export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY tracks/
    DESTINATION share/${PROJECT_NAME}/f1_tracks
    FILES_MATCHING PATTERN "*.track"
)
install(DIRECTORY tracks/
    DESTINATION share/${PROJECT_NAME}/f1_tracks
    FILES_MATCHING PATTERN "*.json"
)


if(ament_cmake_FOUND)
    message(STATUS "Ament was found. Building a ROS2-style cmake package")
    ament_export_targets(${PROJECT_NAME}-export)
    if (NOT USING_VJOY)
        set(USING_VJOY OFF)
    endif()
    ament_package(CONFIG_EXTRAS cmake/find_deps.cmake.in)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/windows_stuff.cmake DESTINATION share/${PROJECT_NAME}/cmake/)
else()
    install(EXPORT ${PROJECT_NAME}-export
    FILE
    ${PROJECT_NAME}Targets.cmake
    NAMESPACE
        f1_datalogger::
    DESTINATION
        cmake
    )
    if (NOT USING_VJOY)
        set(USING_VJOY OFF)
    endif()
    configure_file(${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)

    install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake DESTINATION cmake/)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/windows_stuff.cmake DESTINATION cmake/f1_datalogger/)
    install(FILES ${source_scripts} DESTINATION share/${PROJECT_NAME})


endif()
