
if(MSVC)
	find_package(Eigen3 REQUIRED)
	set(controllers_SRCS
		pure_pursuit_controller.cpp
		vjoy_interface.cpp
	)
else()
	message(FATAL "Interface to control F1 2017 is only supported on Windows.")
endif()
add_library(${PROJECT_NAME}_controllers ${controllers_SRCS})
target_link_libraries(${PROJECT_NAME}_controllers
${PROJECT_NAME}
${PROJECT_NAME}_alglib
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_image_logging_common
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
${Boost_LIBRARIES}
${screen_capture_lite_LIBRARIES}
${screen_capture_lite_PLATFORM_LIBS}
${vJoy++_LIBRARIES}
Eigen3::Eigen
)
target_include_directories(${PROJECT_NAME}_controllers PRIVATE ${vJoy++_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${THIRD_PARTY_REPOS}/libIntegrate/src)
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