if(f1_datalogger_DIR)
    message(STATUS "Using manually specified f1_datalogger_DIR: ${f1_datalogger_DIR}")
elseif(DEFINED ENV{f1_datalogger_DIR}) 
    set(f1_datalogger_DIR $ENV{f1_datalogger_DIR})
    message(STATUS "Using environment f1_datalogger_DIR: ${f1_datalogger_DIR}")
else()
    message(STATUS "Looking for f1_datalogger in the local ament prefix path")
endif()

set(BOOST_REQUIRED_COMPONENTS
date_time
filesystem
program_options
regex
system
thread
)
find_package(Boost COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
if(Boost_FOUND)
    message(STATUS "Found boost via cmake config at ${Boost_DIR}")
else()
    set(Boost_USE_STATIC_LIBS OFF)
    find_package(Boost COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
    if(NOT Boost_FOUND)
        set(Boost_USE_STATIC_LIBS ON)
        find_package(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
    endif(NOT Boost_FOUND)
endif(Boost_FOUND)

if(${Boost_VERSION} VERSION_GREATER_EQUAL 1.67.0)
    list(APPEND Boost_LIBRARIES Boost::headers)
else()
    include_directories(${Boost_INCLUDE_DIRS})
endif(${Boost_VERSION} VERSION_GREATER_EQUAL 1.67.0)
message(STATUS "Using Boost components: ${Boost_LIBRARIES}")

if(WIN32)
    find_package(Armadillo CONFIG REQUIRED)
else()
    find_package(Armadillo REQUIRED)
endif()

find_package(Protobuf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tbb REQUIRED)
find_package(f1_datalogger REQUIRED)
