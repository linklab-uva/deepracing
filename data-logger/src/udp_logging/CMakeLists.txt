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
)
add_library(${PROJECT_NAME}_udp_logging ${${PROJECT_NAME}_udp_logging_src})


set(${PROJECT_NAME}_udp_logging_common_src
common/multi_threaded_udp_handler.cpp
)
add_library( ${PROJECT_NAME}_udp_logging_common ${${PROJECT_NAME}_udp_logging_common_src} )
target_link_libraries( ${PROJECT_NAME}_udp_logging_common ${PROJECT_NAME}_udp_logging ${TBB_IMPORTED_TARGETS} ${Boost_LIBRARIES} )

set(install_targets
#executables.
${PROJECT_NAME}_simulated_telemetry_server
#libraries.
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)
