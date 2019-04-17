
set(${PROJECT_NAME}_utils_src
utils/opencv_utils.cpp
)
add_library(${PROJECT_NAME}_image_utils ${${PROJECT_NAME}_utils_src})
target_link_libraries(${PROJECT_NAME}_image_utils ${OpenCV_LIBRARIES} ${screen_capture_lite_LIBRARIES})


set(${PROJECT_NAME}_image_logging_src
f1_framegrab_manager.cpp
)
add_library(${PROJECT_NAME}_image_logging ${${PROJECT_NAME}_image_logging_src})
target_link_libraries(${PROJECT_NAME}_image_logging ${PROJECT_NAME}_image_utils ${screen_capture_lite_PLATFORM_LIBS})


set(${PROJECT_NAME}_image_logging_common_src
common/multi_threaded_framegrab_handler.cpp
)
add_library(${PROJECT_NAME}_image_logging_common ${${PROJECT_NAME}_image_logging_common_src})
target_link_libraries(${PROJECT_NAME}_image_logging_common ${PROJECT_NAME}_image_logging ${PROJECT_NAME}_image_utils ${PROJECT_NAME}_proto ${TBB_IMPORTED_TARGETS} ${Boost_LIBRARIES})

set(install_targets
#libraries.
${PROJECT_NAME}_image_logging_common
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_image_utils
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)

