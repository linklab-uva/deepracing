project(f1_datalogger)
cmake_minimum_required (VERSION 3.14 FATAL_ERROR)

set(cxx_standard 17)
set(CMAKE_CXX_STANDARD ${cxx_standard})


if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
else()
    set_property(GLOBAL PROPERTY USE_FOLDERS OFF)
    option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "..." FORCE)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(THIRD_PARTY_REPOS ${CMAKE_SOURCE_DIR}/third_party)
include_directories(${EIGEN3_INCLUDE_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(Eigen3)
if((NOT EIGEN3_FOUND))
  message("Could not find Eigen3. Using Eigen3 packaged with this repo.")
  include(add_prepackaged_eigen)
elseif(${EIGEN3_VERSION_STRING} VERSION_LESS "3.3.90")
  message("Found unacceptable eigen version ${EIGEN3_VERSION_STRING} at ${EIGEN3_ROOT_DIR}. Using Eigen3 packaged with this repo.")
  include(add_prepackaged_eigen)
endif()
include_directories(${EIGEN3_INCLUDE_DIRS})

option(BUILD_DATALOGGER "Builds the datalogger libraries and executables. If set to OFF, only builds the ability to interact with the protobuf datasets" ON)
option(GUI_TOOLS "Attempt to build the GUI tools" ON)
if(BUILD_DATALOGGER)
    if(MSVC)
		set(DIRECTXK_DIR ${THIRD_PARTY_REPOS}/DirectXTK)
		add_subdirectory(${DIRECTXK_DIR})
		set(DIRECTXT_INCLUDE_DIR ${DIRECTXK_DIR}/Inc)
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
headers
program_options
regex
system
thread
)
find_package( Boost 1.6 REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS} )
foreach(boost_component_ ${BOOST_REQUIRED_COMPONENTS})
	list(APPEND Boost_LIBRARIES Boost::${boost_component_})
endforeach()
message(STATUS "Using Boost components: ${Boost_LIBRARIES}")



find_package( OpenCV REQUIRED )
find_package(Qt5Widgets CONFIG)
find_package(Qt5Core CONFIG)
find_package(Qt5Gui CONFIG)
set(_gotqt ${Qt5Widgets_FOUND} AND ${Qt5Core_FOUND} AND ${Qt5Gui_FOUND})


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
	message(STATUS "Got TBB targets: ${TBB_IMPORTED_TARGETS}")
endif()


include_directories( include )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIR} )

add_subdirectory( proto/${PROJECT_NAME}/proto )
include_directories(${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR})
add_subdirectory(src/controllers)
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
	option(WITH_ARMA "Use Armadillo library for matrix loading/storing" ON)
	if(WITH_ARMA)
		include(include_arma)
	endif()
    add_subdirectory( src/udp_logging )
    add_subdirectory( src/image_logging )
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
if(${_gotqt} AND GUI_TOOLS)
    add_subdirectory(src/guitools)
endif()


install(DIRECTORY proto/
DESTINATION share/${PROJECT_NAME}/proto
FILES_MATCHING PATTERN "*.proto"
)
