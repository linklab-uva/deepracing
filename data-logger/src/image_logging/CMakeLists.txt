
set(${PROJECT_NAME}_utils_src
utils/opencv_utils.cpp
)
add_library(${PROJECT_NAME}_image_utils ${${PROJECT_NAME}_utils_src})
target_link_libraries(${PROJECT_NAME}_image_utils ${OpenCV_LIBRARIES} ${screen_capture_lite_LIBRARIES} ${PROJECT_NAME}_proto)


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
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
target_link_libraries(${PROJECT_NAME}_image_logging_common stdc++fs )
endif()


add_executable(${PROJECT_NAME}_image_server services/ImageServer.cpp)
if(WIN32)
target_compile_definitions(${PROJECT_NAME}_proto PRIVATE _WIN32_WINNT=${win32_ver})
target_compile_definitions(${PROJECT_NAME}_image_server PRIVATE _WIN32_WINNT=${win32_ver})
endif()
target_include_directories(${PROJECT_NAME}_image_server PRIVATE ${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}_image_server ${PROJECT_NAME}_proto ${PROJECT_NAME}_image_utils ${screen_capture_lite_LIBRARIES} )
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
target_link_libraries(${PROJECT_NAME}_image_server stdc++fs ${screen_capture_lite_PLATFORM_LIBS})
endif()

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

