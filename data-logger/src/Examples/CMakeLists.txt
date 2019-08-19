
set(${PROJECT_NAME}_generate_vjoy_calibration_srcfile generate_vjoy_calibration.cpp)
set(${PROJECT_NAME}_test_vjoy_calibration_srcfile test_vjoy_calibration.cpp)
set(${PROJECT_NAME}_replay_dataset_srcfile replay_dataset.cpp)
set(${PROJECT_NAME}_go_in_circle_srcfile go_in_circle.cpp)
add_executable(${PROJECT_NAME}_go_in_circle
    ${${PROJECT_NAME}_go_in_circle_srcfile}
)
target_link_libraries(${PROJECT_NAME}_go_in_circle
    ${PROJECT_NAME}
    ${PROJECT_NAME}_alglib
	${PROJECT_NAME}_controllers
    ${PROJECT_NAME}_image_logging
    ${PROJECT_NAME}_image_logging_common
    ${PROJECT_NAME}_udp_logging
    ${PROJECT_NAME}_udp_logging_common
	${PROJECT_NAME}_post_processing
    ${Boost_LIBRARIES}
	${screen_capture_lite_LIBRARIES}
	${screen_capture_lite_PLATFORM_LIBS}
	${TBB_IMPORTED_TARGETS}
)

add_executable(${PROJECT_NAME}_replay_dataset
    ${${PROJECT_NAME}_replay_dataset_srcfile}
)
target_link_libraries(${PROJECT_NAME}_replay_dataset
    ${PROJECT_NAME}
    ${PROJECT_NAME}_alglib
    ${PROJECT_NAME}_image_logging
    ${PROJECT_NAME}_image_logging_common
    ${PROJECT_NAME}_udp_logging
    ${PROJECT_NAME}_udp_logging_common
	${PROJECT_NAME}_post_processing
    ${Boost_LIBRARIES}
	${screen_capture_lite_LIBRARIES}
	${screen_capture_lite_PLATFORM_LIBS}
	${TBB_IMPORTED_TARGETS}
	${PROJECT_NAME}_controllers
)
if(MSVC)
    find_package(vJoy++ REQUIRED)
	add_executable(${PROJECT_NAME}_generate_vjoy_calibration
	${${PROJECT_NAME}_generate_vjoy_calibration_srcfile}
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
		${PROJECT_NAME}_controllers
	)
	target_include_directories(${PROJECT_NAME}_generate_vjoy_calibration PRIVATE ${vJoy++_INCLUDE_DIR})
	add_executable(${PROJECT_NAME}_test_vjoy_calibration
	${${PROJECT_NAME}_test_vjoy_calibration_srcfile}
	)
	target_link_libraries(${PROJECT_NAME}_test_vjoy_calibration
		${PROJECT_NAME}
		${PROJECT_NAME}_controllers
		${PROJECT_NAME}_image_logging
		${PROJECT_NAME}_image_logging_common
		${PROJECT_NAME}_udp_logging
		${PROJECT_NAME}_udp_logging_common
		${Boost_LIBRARIES}
		${screen_capture_lite_LIBRARIES}
		${screen_capture_lite_PLATFORM_LIBS}
	)
	target_include_directories(${PROJECT_NAME}_test_vjoy_calibration PRIVATE ${vJoy++_INCLUDE_DIR})
	set_target_properties(${PROJECT_NAME}_test_vjoy_calibration ${PROJECT_NAME}_generate_vjoy_calibration PROPERTIES FOLDER vjoy)
endif()

	
add_executable(${PROJECT_NAME}_generate_sine_waves
    generate_sine_waves.cpp
)
target_link_libraries(${PROJECT_NAME}_generate_sine_waves
    ${PROJECT_NAME}
    ${PROJECT_NAME}_alglib
	${PROJECT_NAME}_controllers
    ${PROJECT_NAME}_image_logging
    ${PROJECT_NAME}_image_logging_common
    ${PROJECT_NAME}_udp_logging
    ${PROJECT_NAME}_udp_logging_common
	${PROJECT_NAME}_post_processing
    ${Boost_LIBRARIES}
	${screen_capture_lite_LIBRARIES}
	${screen_capture_lite_PLATFORM_LIBS}
	${TBB_IMPORTED_TARGETS}
	${YAML_CPP_LIBRARIES}
)
set_target_properties(${PROJECT_NAME}_go_in_circle ${PROJECT_NAME}_replay_dataset ${PROJECT_NAME}_generate_sine_waves PROPERTIES FOLDER vjoy)

add_executable(${PROJECT_NAME}_pure_pursuit_control
    pure_pursuit_control.cpp
)
target_link_libraries(${PROJECT_NAME}_pure_pursuit_control
    ${PROJECT_NAME}
    ${PROJECT_NAME}_alglib
	${PROJECT_NAME}_controllers
    ${PROJECT_NAME}_image_logging
    ${PROJECT_NAME}_image_logging_common
	${PROJECT_NAME}_post_processing
    ${PROJECT_NAME}_udp_logging
    ${PROJECT_NAME}_udp_logging_common
    ${Boost_LIBRARIES}
	${screen_capture_lite_LIBRARIES}
	${screen_capture_lite_PLATFORM_LIBS}
	${TBB_IMPORTED_TARGETS}
	${YAML_CPP_LIBRARIES}
)
set_target_properties(${PROJECT_NAME}_pure_pursuit_control PROPERTIES FOLDER vjoy)
get_target_property(PP_OUTPUT_DIR ${PROJECT_NAME}_pure_pursuit_control RUNTIME_OUTPUT_DIRECTORY)
file(GLOB TRACKFILES
	"${CMAKE_SOURCE_DIR}/tracks/*.track"
)
message(STATUS "Output trackfile dir: ${PP_OUTPUT_DIR}")
file(COPY ${TRACKFILES} DESTINATION ${PP_OUTPUT_DIR})


add_executable(${PROJECT_NAME}_cv_viewer_example
cv_viewer.cpp
)
target_link_libraries(${PROJECT_NAME}_cv_viewer_example
${PROJECT_NAME}
${PROJECT_NAME}_eigen_utils
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_image_logging_common
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
${screen_capture_lite_LIBRARIES}
${screen_capture_lite_PLATFORM_LIBS}
${Boost_LIBRARIES}
${TBB_IMPORTED_TARGETS}
Eigen3::Eigen
)
target_include_directories(${PROJECT_NAME}_cv_viewer_example PRIVATE ${DIRECTXT_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME}_cv_viewer_example PROPERTIES FOLDER examples)

if(MSVC)
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
	directxtk
	)
	target_include_directories(${PROJECT_NAME}_multithreaded_capture_example PRIVATE ${DIRECTXT_INCLUDE_DIR})
	set_target_properties(${PROJECT_NAME}_multithreaded_capture_example PROPERTIES FOLDER examples)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_link_libraries(${PROJECT_NAME}_replay_dataset stdc++fs )
    target_link_libraries(${PROJECT_NAME}_generate_sine_waves stdc++fs )
    target_link_libraries(${PROJECT_NAME}_pure_pursuit_control stdc++fs )
endif()

set(install_targets
#executables.
${PROJECT_NAME}_cv_viewer_example
)
if(WITH_VJOY)
    list(APPEND install_targets ${PROJECT_NAME}_pure_pursuit_control)
    list(APPEND install_targets ${PROJECT_NAME}_test_vjoy_calibration)
    list(APPEND install_targets ${PROJECT_NAME}_generate_vjoy_calibration)
    list(APPEND install_targets ${PROJECT_NAME}_replay_dataset)
	list(APPEND install_targets ${PROJECT_NAME}_generate_sine_waves)
	list(APPEND install_targets ${PROJECT_NAME}_multithreaded_capture_example)
	install(DIRECTORY ${CMAKE_SOURCE_DIR}/tracks/ DESTINATION bin
        FILES_MATCHING PATTERN "*.track")
endif()
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)

