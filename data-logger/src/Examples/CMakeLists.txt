option(WITH_VJOY "Compile vJoy bindings. ONLY SUPPORTED ON WINDOWS" OFF)
if(WITH_VJOY)
    if(NOT MSVC)
        message(FATAL_ERROR "vJoy bindings only work on windows.")
    endif()
    find_package(vJoy++ REQUIRED)
    message(STATUS "vJoy++ include dir: ${vJoy++_INCLUDE_DIR}")
    message(STATUS "vJoy++ libraries: ${vJoy++_LIBRARIES}")
	add_executable(${PROJECT_NAME}_generate_vjoy_calibration
    generate_vjoy_calibration.cpp
    )
    target_link_libraries(${PROJECT_NAME}_generate_vjoy_calibration
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
    target_include_directories(${PROJECT_NAME}_generate_vjoy_calibration PRIVATE ${vJoy++_INCLUDE_DIR})

    add_executable(${PROJECT_NAME}_test_vjoy_calibration
    test_vjoy_calibration.cpp
    )
    target_link_libraries(${PROJECT_NAME}_test_vjoy_calibration
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
    target_include_directories(${PROJECT_NAME}_test_vjoy_calibration PRIVATE ${vJoy++_INCLUDE_DIR})
endif()

add_executable(${PROJECT_NAME}_cv_viewer_example
cv_viewer.cpp
)
target_link_libraries(${PROJECT_NAME}_cv_viewer_example
${PROJECT_NAME}
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_image_logging_common
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
${screen_capture_lite_LIBRARIES}
${screen_capture_lite_PLATFORM_LIBS}
${Boost_LIBRARIES}
${TBB_IMPORTED_TARGETS}
)


add_executable(${PROJECT_NAME}_multithreaded_capture_example
multi_threaded_capture.cpp
)
target_link_libraries(${PROJECT_NAME}_multithreaded_capture_example
${PROJECT_NAME}
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_image_logging_common
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
${PROJECT_NAME}_proto
${screen_capture_lite_LIBRARIES}
${screen_capture_lite_PLATFORM_LIBS}
${Boost_LIBRARIES}
${TBB_IMPORTED_TARGETS}
${YAML_CPP_LIBRARIES}
)

set(install_targets
#executables.
${PROJECT_NAME}_multithreaded_capture_example
${PROJECT_NAME}_cv_viewer_example
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)
