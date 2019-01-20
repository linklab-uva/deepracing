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
add_dependencies(${PROJECT_NAME}_image_logging protocopy)
target_link_libraries(${PROJECT_NAME}_udp_logging ${PROJECT_NAME}_utils ${PROJECT_NAME}_proto ${TBB_IMPORTED_TARGETS} ${Boost_LIBRARIES})

set(install_targets
#executables.
${PROJECT_NAME}_simulated_telemetry_server
#libraries.
${PROJECT_NAME}_udp_logging
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)
