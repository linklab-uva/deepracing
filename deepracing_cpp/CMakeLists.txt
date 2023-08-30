set(_required_cmake 3.22)
cmake_minimum_required (VERSION ${_required_cmake} FATAL_ERROR)
cmake_policy(VERSION ${_required_cmake})
project(deepracing_cpp CXX)

find_package(ament_cmake QUIET)
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif()

add_library(headers INTERFACE)
target_include_directories(headers
INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
)
set(targets_to_install)
list(APPEND targets_to_install headers)

find_package(PCL CONFIG REQUIRED)
message(STATUS "Found PCL at ${PCL_DIR}")
add_library(pcltypes SHARED src/pcl_types.cpp)
target_link_libraries(pcltypes PUBLIC headers PRIVATE ${PCL_LIBRARIES})
target_include_directories(pcltypes PRIVATE ${PCL_INCLUDE_DIRS})
set_target_properties(pcltypes PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_pcltypes)
if(WIN32)
    set_target_properties(pcltypes PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    target_compile_definitions(pcltypes PUBLIC "WIN32_LEAN_AND_MEAN" "NOGDI")
endif()
list(APPEND targets_to_install pcltypes)
set_target_properties(pcltypes PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_pcltypes)


find_package(Eigen3 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
message(STATUS "Found yaml-cpp at ${yaml-cpp_DIR}")
add_library(utils SHARED src/track_map.cpp src/utils.cpp)
target_link_libraries(utils PUBLIC headers pcltypes yaml-cpp Eigen3::Eigen PRIVATE ${PCL_LIBRARIES})
target_include_directories(utils PRIVATE ${PCL_INCLUDE_DIRS})
set_target_properties(utils PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_utils)
if(WIN32)
    target_compile_definitions(utils PRIVATE DEEPRACING_CPP_BUILDING_DLL)
endif()
list(APPEND targets_to_install utils)
set_target_properties(utils PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_utils CXX_STANDARD 17 CXX_STANDARD_REQUIRED TRUE)


set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf QUIET)
find_package(absl CONFIG QUIET)
find_package(glog CONFIG QUIET)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed Development.Module)
find_package(pybind11 CONFIG QUIET)
set(_waymo_tensorflow_conversion_ FALSE)
if(Protobuf_FOUND AND absl_FOUND AND glog_FOUND AND pybind11_FOUND)
    add_subdirectory(waymo_tensorflow)
    set(_waymo_tensorflow_conversion_ TRUE)
endif()

install(DIRECTORY include/deepracing/
    DESTINATION include/deepracing
    FILES_MATCHING PATTERN "*.h*"
)
install(TARGETS ${targets_to_install}
    EXPORT ${PROJECT_NAME}_export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(ament_cmake_FOUND)
    set(${PROJECT_NAME}_BUILT_WITH_AMENT TRUE)
    ament_export_targets(${PROJECT_NAME}_export)
    ament_package(CONFIG_EXTRAS cmake/find_deps.cmake.in)
else()
    set(${PROJECT_NAME}_BUILT_WITH_AMENT FALSE)
    install(EXPORT ${PROJECT_NAME}_export
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION cmake
    )
    configure_file(${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
    configure_file(cmake/find_deps.cmake.in find_deps.cmake @ONLY)
    install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake DESTINATION cmake/)
    install(FILES ${CMAKE_BINARY_DIR}/find_deps.cmake DESTINATION cmake/)
endif()