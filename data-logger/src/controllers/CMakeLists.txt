
find_package(Eigen3 REQUIRED)

set(controllers_SRCS
    pure_pursuit_controller.cpp
)

add_library(${PROJECT_NAME}_controllers ${controllers_SRCS})
target_link_libraries(${PROJECT_NAME}_controllers
${PROJECT_NAME}
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_image_logging_common
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
${Boost_LIBRARIES}
${screen_capture_lite_LIBRARIES}
${screen_capture_lite_PLATFORM_LIBS}
${vJoy++_LIBRARIES}
)
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