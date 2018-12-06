
set(${PROJECT_NAME}_utils_src
utils/opencv_utils.cpp
#utils/screencapture_lite_utils.cpp
)
add_library(${PROJECT_NAME}_utils ${${PROJECT_NAME}_utils_src})
target_link_libraries(${PROJECT_NAME}_utils ${OpenCV_LIBRARIES} ${screen_capture_lite_LIBRARIES})


set(${PROJECT_NAME}_image_logging_src
f1_framegrab_manager.cpp
common/multi_threaded_framegrab_handler.cpp
)
add_library(${PROJECT_NAME}_image_logging ${${PROJECT_NAME}_image_logging_src})
target_link_libraries(${PROJECT_NAME}_image_logging ${PROJECT_NAME}_utils tbb)


