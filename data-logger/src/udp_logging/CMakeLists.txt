include(FindThreads)
add_executable(${PROJECT_NAME}_simulated_telemetry_server
simulated_telemetry_server.cpp
)
target_link_libraries(${PROJECT_NAME}_simulated_telemetry_server
${Boost_LIBRARIES}
${CMAKE_THREAD_LIBS_INIT}
)

set(${PROJECT_NAME}_udp_logging_src
f1_datagrab_manager.cpp
common/multi_threaded_udp_handler.cpp
)

add_library(${PROJECT_NAME}_udp_logging ${${PROJECT_NAME}_udp_logging_src})
target_include_directories(${PROJECT_NAME}_udp_logging PRIVATE ${ProtobufIncludePath} ${PROTOBUF_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}_udp_logging ${PROJECT_NAME}_utils ${PROJECT_NAME}_proto ${TBB_IMPORTED_TARGETS} ${Boost_LIBRARIES})
