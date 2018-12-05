
set(${PROJECT_NAME}_opencv_utils_src
utils/opencv_utils.cpp
)
add_library(${PROJECT_NAME}_opencv_utils ${${PROJECT_NAME}_opencv_utils_src})
target_link_libraries(${PROJECT_NAME}_opencv_utils ${OpenCV_LIBRARIES})


set(${PROJECT_NAME}_image_logging_src
f1_framegrab_manager.cpp
)
add_library(${PROJECT_NAME}_image_logging ${${PROJECT_NAME}_image_logging_src})
target_link_libraries(${PROJECT_NAME}_image_logging ${OpenCV_LIBRARIES} ${PROJECT_NAME}_opencv_utils)


