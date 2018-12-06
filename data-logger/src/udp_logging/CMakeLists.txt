add_executable(${PROJECT_NAME}_simulated_telemetry_server
simulated_telemetry_server.cpp
)
target_link_libraries(${PROJECT_NAME}_simulated_telemetry_server
${Boost_LIBRARIES}
pthread
)

set(${PROJECT_NAME}_udp_logging_src
f1_datagrab_manager.cpp
)

add_library(${PROJECT_NAME}_udp_logging ${${PROJECT_NAME}_udp_logging_src})