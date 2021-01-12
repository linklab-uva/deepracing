
set(${PROJECT_NAME}_utils_src
utils/opencv_utils.cpp
)
add_library(${PROJECT_NAME}_image_utils ${${PROJECT_NAME}_utils_src})
target_link_libraries(${PROJECT_NAME}_image_utils ${OpenCV_LIBRARIES} ${screen_capture_lite_LIBRARIES} ${PROJECT_NAME}_proto)
target_include_directories(${PROJECT_NAME}_image_utils 
PRIVATE
${PROTO_INCLUDE_DIR}
)
target_compile_definitions(${PROJECT_NAME}_image_utils PRIVATE F1_DATALOGGER_IMAGE_LOGGING_BUILDING_DLL)

if(WIN32 AND GUI_TOOLS)
    add_subdirectory(winrtcapture)
    add_subdirectory(qtcapture)
endif()
set(USE_WINRT_GRAPHICS OFF)
configure_file(graphics_capture_config.h.in ${CMAKE_BINARY_DIR}/graphics_config/f1_datalogger/image_logging/graphics_capture_config.h @ONLY)
if(USE_WINRT_GRAPHICS)
    add_library(${PROJECT_NAME}_image_logging windowsrt/f1_framegrab_manager.cpp)
    target_link_libraries(${PROJECT_NAME}_image_logging winrt_capture_lib)
    target_compile_definitions(${PROJECT_NAME}_image_logging PUBLIC USE_WINRT_GRAPHICS)
else()
    add_library(${PROJECT_NAME}_image_logging screencapturelite/f1_framegrab_manager.cpp)
    target_link_libraries(${PROJECT_NAME}_image_logging ${screen_capture_lite_PLATFORM_LIBS})
endif()
target_link_libraries(${PROJECT_NAME}_image_logging ${PROJECT_NAME}_image_utils)
target_include_directories(${PROJECT_NAME}_image_logging 
    PRIVATE
    ${PROTO_INCLUDE_DIR}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/graphics_config>
        $<INSTALL_INTERFACE:include>
)
target_compile_definitions(${PROJECT_NAME}_image_logging PRIVATE F1_DATALOGGER_IMAGE_LOGGING_BUILDING_DLL)
install(FILES ${CMAKE_BINARY_DIR}/graphics_config/f1_datalogger/image_logging/graphics_capture_config.h
    DESTINATION include/f1_datalogger/image_logging
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
${Boost_TARGETS}
)
if (NOT HAS_STDCXXFS)
    target_compile_definitions(${PROJECT_NAME}_image_logging_common PRIVATE BOOST_FILESYSTEM=1)
endif (NOT HAS_STDCXXFS)
target_link_libraries(${PROJECT_NAME}_image_logging_common ${image_logging_libs} )
target_include_directories(${PROJECT_NAME}_image_logging_common 
PRIVATE
${PROTO_INCLUDE_DIR}
)
target_compile_definitions(${PROJECT_NAME}_image_logging_common PRIVATE F1_DATALOGGER_IMAGE_LOGGING_BUILDING_DLL)


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

