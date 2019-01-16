add_executable(${PROJECT_NAME}_cv_viewer_example
cv_viewer.cpp
)
target_link_libraries(${PROJECT_NAME}_cv_viewer_example
${PROJECT_NAME}
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_udp_logging
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
${PROJECT_NAME}_udp_logging
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
