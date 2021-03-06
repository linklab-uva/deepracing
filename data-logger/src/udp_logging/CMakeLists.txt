include(FindThreads)
add_executable(${PROJECT_NAME}_simulated_telemetry_server
simulated_telemetry_server.cpp
)
target_link_libraries(${PROJECT_NAME}_simulated_telemetry_server
${Boost_TARGETS}
${CMAKE_THREAD_LIBS_INIT}
)


set(${PROJECT_NAME}_udp_logging_src
f1_datagrab_manager.cpp
)
add_library(${PROJECT_NAME}_udp_logging ${${PROJECT_NAME}_udp_logging_src})
target_link_libraries(${PROJECT_NAME}_udp_logging ${Boost_TARGETS} )
target_compile_definitions(${PROJECT_NAME}_udp_logging PRIVATE F1_DATALOGGER_UDP_LOGGING_BUILDING_DLL)

set(${PROJECT_NAME}_udp_utils_src
utils/udp_stream_utils.cpp
)
add_library( ${PROJECT_NAME}_udp_utils ${${PROJECT_NAME}_udp_utils_src} )
target_link_libraries(${PROJECT_NAME}_udp_utils 
${PROJECT_NAME}_udp_logging 
${PROJECT_NAME}_proto 
${Boost_TARGETS}
)
target_compile_definitions(${PROJECT_NAME}_udp_utils PRIVATE F1_DATALOGGER_UDP_LOGGING_BUILDING_DLL)



set(${PROJECT_NAME}_eigen_utils_src
utils/eigen_utils.cpp
)
add_library( ${PROJECT_NAME}_eigen_utils ${${PROJECT_NAME}_eigen_utils_src} )
if(WITH_ARMA)
	target_compile_definitions(${PROJECT_NAME}_eigen_utils  PRIVATE USE_ARMADILLO)
    target_include_directories(${PROJECT_NAME}_eigen_utils PRIVATE ${ARMADILLO_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME}_eigen_utils ${ARMADILLO_LIBRARIES})
endif()
target_link_libraries( ${PROJECT_NAME}_eigen_utils Eigen3::Eigen ${Boost_TARGETS} ${PROJECT_NAME}_proto ${PROJECT_NAME}_udp_utils )
target_include_directories(${PROJECT_NAME}_eigen_utils PRIVATE ${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR})
target_compile_definitions(${PROJECT_NAME}_eigen_utils PRIVATE F1_DATALOGGER_UDP_LOGGING_BUILDING_DLL)


get_target_property(protoincludes ${PROJECT_NAME}_proto INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Includes inherited from proto library: ${protoincludes}")
set(${PROJECT_NAME}_udp_logging_common_src
common/multi_threaded_udp_handler.cpp
common/multi_threaded_udp_handler_2018.cpp
common/measurement_handler.cpp
common/measurement_handler_2018.cpp
common/rebroadcast_handler_2018.cpp
)
add_library( ${PROJECT_NAME}_udp_logging_common ${${PROJECT_NAME}_udp_logging_common_src} )
set(udp_logging_common_libs
${PROJECT_NAME}_udp_logging 
${PROJECT_NAME}_udp_utils
${PROJECT_NAME}_proto
${TBB_IMPORTED_TARGETS} 
${Boost_TARGETS} 
)
if (NOT HAS_STDCXXFS)
    target_compile_definitions(${PROJECT_NAME}_udp_logging_common PRIVATE BOOST_FILESYSTEM=1)
endif (NOT HAS_STDCXXFS)
target_link_libraries( ${PROJECT_NAME}_udp_logging_common 
PUBLIC
${udp_logging_common_libs}
)
target_compile_definitions(${PROJECT_NAME}_udp_logging_common PRIVATE F1_DATALOGGER_UDP_LOGGING_BUILDING_DLL)

set(install_targets
#executables.
${PROJECT_NAME}_simulated_telemetry_server
#libraries.
${PROJECT_NAME}_eigen_utils
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
${PROJECT_NAME}_udp_utils
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)
