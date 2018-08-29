project(f1_datalogger)
cmake_minimum_required(VERSION 2.7)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
find_package( Boost COMPONENTS chrono filesystem math_tr1f math_tr1l program_options REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})
include_directories(include)
link_directories(${Boost_LIBRARY_DIR_RELEASE})
option(BUILD_DLLS "Generate Dynamic Link Libraries (DLLs)" OFF)
option(BUILD_PYTHON_BINDING "Generate pybind11 bindings" OFF)



IF(BUILD_DLLS)
set(BUILD_SHARED_LIBS true)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS true)
ELSE()
set(BUILD_SHARED_LIBS false)
ENDIF()
set(SOCKET_LIBRARY ws2_32)
add_definitions(-std=c++11)
add_library(${PROJECT_NAME}_data_capture
src/simple_udp_listener
src/screen_video_capture.cpp
src/simple_screen_listener.cpp
)
target_link_libraries(${PROJECT_NAME}_data_capture
${OpenCV_LIBS}
${SOCKET_LIBRARY}
debug ${Boost_CHRONO_LIBRARY_DEBUG}
optimized ${Boost_CHRONO_LIBRARY_RELEASE}
)

add_library(${PROJECT_NAME}_gsoap_binding
src/stdsoap2.cpp
src/deepf1_gsoap_env.cpp 
src/deepf1_gsoapC.cpp
src/nsmap.cpp
)

add_library(${PROJECT_NAME}_gsoap_conversions
src/gsoap_conversions.cpp
)
target_link_libraries(${PROJECT_NAME}_gsoap_conversions
${PROJECT_NAME}_gsoap_binding
)
add_library(${PROJECT_NAME}_math_utils
src/alglib/alglibinternal.cpp
src/alglib/alglibmisc.cpp
src/alglib/ap.cpp
src/alglib/dataanalysis.cpp
src/alglib/diffequations.cpp
src/alglib/fasttransforms.cpp
src/alglib/integration.cpp
src/alglib/interpolation.cpp
src/alglib/linalg.cpp
src/alglib/optimization.cpp
src/alglib/solvers.cpp
src/alglib/specialfunctions.cpp
src/alglib/statistics.cpp
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
#${PROJECT_NAME}_math_utils
debug ${Boost_CHRONO_LIBRARY_DEBUG}
optimized ${Boost_CHRONO_LIBRARY_RELEASE}
debug ${Boost_FILESYSTEM_LIBRARY_DEBUG}
optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
)

add_executable(${PROJECT_NAME}_simulated_telmetry_server src/simulated_telemetry_server.cpp)
target_link_libraries(${PROJECT_NAME}_simulated_telmetry_server
${PROJECT_NAME}_data_capture
${OpenCV_LIBS}
debug ${Boost_CHRONO_LIBRARY_DEBUG}
optimized ${Boost_CHRONO_LIBRARY_RELEASE}
debug ${Boost_FILESYSTEM_LIBRARY_DEBUG}
optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
)

add_executable(${PROJECT_NAME}_steering_extractor src/steering_extractor.cpp)
target_link_libraries(${PROJECT_NAME}_steering_extractor 
${PROJECT_NAME}_math_utils
${PROJECT_NAME}_data_capture
${PROJECT_NAME}_gsoap_binding
debug ${Boost_FILESYSTEM_LIBRARY_DEBUG}
optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
)

add_executable(${PROJECT_NAME}_common_extractor src/common_extractor.cpp)
target_link_libraries(${PROJECT_NAME}_common_extractor
${PROJECT_NAME}_math_utils
${PROJECT_NAME}_data_capture
${PROJECT_NAME}_gsoap_binding
debug ${Boost_FILESYSTEM_LIBRARY_DEBUG}
optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE}
)

set(${PROJECT_NAME}_INSTALL_TARGETS
${PROJECT_NAME}_math_utils
${PROJECT_NAME}_steering_extractor
${PROJECT_NAME}_common_extractor 
${PROJECT_NAME}_simulated_telmetry_server
${PROJECT_NAME}_gsoap_conversions
${PROJECT_NAME}_gsoap_binding
${PROJECT_NAME}_data_capture
${PROJECT_NAME}
)
if(BUILD_PYTHON_BINDING)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)
add_subdirectory(python_binding)
endif()
install(TARGETS ${${PROJECT_NAME}_INSTALL_TARGETS}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/deepf1
        FILES_MATCHING PATTERN "*.h")
