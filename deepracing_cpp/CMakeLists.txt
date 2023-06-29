set(_required_cmake 3.22)
cmake_minimum_required (VERSION ${_required_cmake} FATAL_ERROR)
cmake_policy(VERSION ${_required_cmake})
project(deepracing_cpp CXX)

find_package(ament_cmake QUIET)

add_library(headers INTERFACE)
target_include_directories(headers
INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
)
set(targets_to_install)
list(APPEND targets_to_install headers)

find_package(PCL CONFIG REQUIRED)
add_library(pcltypes SHARED src/pcl_types.cpp)
target_link_libraries(pcltypes PUBLIC headers PRIVATE ${PCL_LIBRARIES})
target_include_directories(pcltypes PUBLIC ${PCL_INCLUDE_DIRS})
set_target_properties(pcltypes PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_pcltypes)
if(WIN32)
    set_target_properties(pcltypes PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    target_compile_definitions(pcltypes PRIVATE )
endif()
list(APPEND targets_to_install pcltypes)
set_target_properties(pcltypes PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_pcltypes)

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
    ament_export_targets(${PROJECT_NAME}_export)
    ament_package(CONFIG_EXTRAS cmake/find_deps.cmake.in)
else()
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