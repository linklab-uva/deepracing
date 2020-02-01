
set(${PROJECT_NAME}_utils_src
utils/opencv_utils.cpp
)
add_library(${PROJECT_NAME}_image_utils ${${PROJECT_NAME}_utils_src})
target_link_libraries(${PROJECT_NAME}_image_utils ${OpenCV_LIBRARIES} ${screen_capture_lite_LIBRARIES} ${PROJECT_NAME}_proto)
target_include_directories(${PROJECT_NAME}_image_utils 
PRIVATE
${PROTO_INCLUDE_DIR}
)



set(${PROJECT_NAME}_image_logging_src
f1_framegrab_manager.cpp
)
add_library(${PROJECT_NAME}_image_logging ${${PROJECT_NAME}_image_logging_src})
target_link_libraries(${PROJECT_NAME}_image_logging ${PROJECT_NAME}_image_utils ${screen_capture_lite_PLATFORM_LIBS})
target_include_directories(${PROJECT_NAME}_image_logging 
PRIVATE
${PROTO_INCLUDE_DIR}
)



set(${PROJECT_NAME}_image_logging_common_src
common/multi_threaded_framegrab_handler.cpp
)
add_library(${PROJECT_NAME}_image_logging_common ${${PROJECT_NAME}_image_logging_common_src})
set(image_logging_libs
${PROJECT_NAME}_image_logging 
${PROJECT_NAME}_image_utils 
${PROJECT_NAME}_proto 
${TBB_IMPORTED_TARGETS} 
${Boost_LIBRARIES}
)
if (
    ( ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 8.1) ) OR
    ( ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 9.0) ) OR
    (${CMAKE_CXX_STANDARD} VERSION_LESS "17")
   )
    target_compile_definitions(${PROJECT_NAME}_image_logging_common PRIVATE BOOST_FILESYSTEM=1)
endif()
target_link_libraries(${PROJECT_NAME}_image_logging_common ${image_logging_libs} )
target_include_directories(${PROJECT_NAME}_image_logging_common 
PRIVATE
${PROTO_INCLUDE_DIR}
)


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

