find_package(Eigen3 REQUIRED)
if(MSVC)
    find_package(vJoy++ REQUIRED)
	set(controllers_SRCS
		pure_pursuit_controller.cpp
		windows/vjoy_interface.cpp
		windows/f1_interface_factory.cpp
	)
	set(additional_controller_libs_ ${vJoy++_LIBRARIES})
	set(additional_controller_includes_ ${vJoy++_INCLUDE_DIR})
else()
	set(controllers_SRCS
		pure_pursuit_controller.cpp
		linux/f1_interface_factory.cpp
	)
endif()
add_library(${PROJECT_NAME}_controllers ${controllers_SRCS})
target_link_libraries(${PROJECT_NAME}_controllers
${PROJECT_NAME}
${PROJECT_NAME}_alglib
${PROJECT_NAME}_eigen_utils
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_image_logging_common
${PROJECT_NAME}_post_processing
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
${Boost_LIBRARIES}
${screen_capture_lite_LIBRARIES}
${screen_capture_lite_PLATFORM_LIBS}
Eigen3::Eigen
${additional_controller_libs_}
)
message(STATUS "Using ${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR} for controller library")
target_include_directories(${PROJECT_NAME}_controllers PRIVATE ${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR} ${additional_controller_includes_})

add_dependencies(${PROJECT_NAME}_controllers ${PROJECT_NAME}_proto)
set_target_properties(${PROJECT_NAME}_controllers
                      PROPERTIES FOLDER vjoy)
set(install_targets
#executables.

#libraries
${PROJECT_NAME}_controllers
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)