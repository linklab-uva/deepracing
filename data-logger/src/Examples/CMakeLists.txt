
set(${PROJECT_NAME}_generate_vjoy_calibration_srcfile generate_vjoy_calibration.cpp)
set(${PROJECT_NAME}_test_vjoy_calibration_srcfile test_vjoy_calibration.cpp)
set(${PROJECT_NAME}_replay_dataset_srcfile replay_dataset.cpp)
set(${PROJECT_NAME}_go_in_circle_srcfile go_in_circle.cpp)


add_executable(${PROJECT_NAME}_cv_viewer_example
cv_viewer.cpp
)
target_link_libraries(${PROJECT_NAME}_cv_viewer_example
${PROJECT_NAME}
${PROJECT_NAME}_eigen_utils
${PROJECT_NAME}_image_logging
${PROJECT_NAME}_image_logging_common
${PROJECT_NAME}_proto
${PROJECT_NAME}_udp_logging
${PROJECT_NAME}_udp_logging_common
${screen_capture_lite_LIBRARIES}
${screen_capture_lite_PLATFORM_LIBS}
${Boost_LIBRARIES}
${TBB_IMPORTED_TARGETS}
Eigen3::Eigen
)
target_include_directories(${PROJECT_NAME}_cv_viewer_example 
PRIVATE
${DIRECTXT_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME}_proto_rebroadcaster
proto_rebroadcaster.cpp
)
target_link_libraries(${PROJECT_NAME}_proto_rebroadcaster
    ${PROJECT_NAME}
    ${PROJECT_NAME}_image_logging
	${PROJECT_NAME}_image_logging_common
	${PROJECT_NAME}_image_utils
	${PROJECT_NAME}_proto
    ${PROJECT_NAME}_udp_logging
    ${PROJECT_NAME}_udp_logging_common
    ${Boost_LIBRARIES}
	${screen_capture_lite_LIBRARIES}
	${screen_capture_lite_PLATFORM_LIBS}
)

set(install_targets
#executables.
${PROJECT_NAME}_proto_rebroadcaster
${PROJECT_NAME}_cv_viewer_example
)

if(MSVC)
	#find_package(vJoy_plusplus REQUIRED)
	set(WITH_VJOY ON)
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
		vjoy::vJoy_plusplus
	)
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
		vjoy::vJoy_plusplus
	)
		
	add_executable(${PROJECT_NAME}_replay_dataset
	${${PROJECT_NAME}_replay_dataset_srcfile}
	)
	target_link_libraries(${PROJECT_NAME}_replay_dataset
		${PROJECT_NAME}
		${PROJECT_NAME}_image_logging
		${PROJECT_NAME}_image_logging_common
		${PROJECT_NAME}_image_utils
		${PROJECT_NAME}_proto
		${PROJECT_NAME}_udp_logging
		${PROJECT_NAME}_udp_logging_common
		${Boost_LIBRARIES}
		${screen_capture_lite_LIBRARIES}
		${screen_capture_lite_PLATFORM_LIBS}
		vjoy::vJoy_plusplus
	)
	add_executable(${PROJECT_NAME}_test_vjoy_lag
	  test_vjoy_lag.cpp
	)
	target_link_libraries(${PROJECT_NAME}_test_vjoy_lag
		${PROJECT_NAME}
		${PROJECT_NAME}_image_logging
		${PROJECT_NAME}_image_logging_common
		${PROJECT_NAME}_image_utils
		${PROJECT_NAME}_proto
		${PROJECT_NAME}_udp_logging
		${PROJECT_NAME}_udp_logging_common
		${Boost_LIBRARIES}
		${screen_capture_lite_LIBRARIES}
		${screen_capture_lite_PLATFORM_LIBS}
		${TBB_IMPORTED_TARGETS} 
		vjoy::vJoy_plusplus
		${YAML_CPP_LIBRARIES}
		yaml-cpp
	)
	if(YAML_CPP_DLL)
		target_compile_definitions(${PROJECT_NAME}_test_vjoy_lag PRIVATE "YAML_CPP_DLL")
	endif()
	list(APPEND install_targets ${PROJECT_NAME}_replay_dataset ${PROJECT_NAME}_test_vjoy_calibration ${PROJECT_NAME}_generate_vjoy_calibration ${PROJECT_NAME}_test_vjoy_lag)
endif()

	
if(${Eigen3_VERSION} VERSION_GREATER_EQUAL "3.3.9")
	add_executable(${PROJECT_NAME}_pure_pursuit_control
		pure_pursuit_control.cpp
	)
	target_link_libraries(${PROJECT_NAME}_pure_pursuit_control
		${PROJECT_NAME}
		${PROJECT_NAME}_controllers
		${PROJECT_NAME}_image_logging
		${PROJECT_NAME}_image_logging_common
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
endif()



if(TARGET ${PROJECT_NAME}_pure_pursuit_control)
   list(APPEND install_targets ${PROJECT_NAME}_pure_pursuit_control)
endif(TARGET ${PROJECT_NAME}_pure_pursuit_control)

option(BUILD_MULTITHREADED_EXAMPLE "" ON)
if(BUILD_MULTITHREADED_EXAMPLE)
	add_executable(${PROJECT_NAME}_multithreaded_capture
	multi_threaded_capture.cpp
	)
	set(multithreaded_capture_libs
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
	yaml-cpp
	)
	if(NOT HAS_STDCXXFS)
		target_compile_definitions(${PROJECT_NAME}_multithreaded_capture PRIVATE BOOST_FILESYSTEM=1)
	endif(NOT HAS_STDCXXFS)

	if(MSVC)
		if(YAML_CPP_DLL)
			target_compile_definitions(${PROJECT_NAME}_multithreaded_capture PRIVATE "YAML_CPP_DLL")
		endif()
		if(WIN10API)
			list(APPEND multithreaded_capture_libs RuntimeObject)
		endif()
	endif()
	target_link_libraries(${PROJECT_NAME}_multithreaded_capture ${multithreaded_capture_libs})
	list(APPEND install_targets ${PROJECT_NAME}_multithreaded_capture)
endif()
if(WITH_VJOY)
	install(DIRECTORY ${CMAKE_SOURCE_DIR}/tracks/ DESTINATION tracks
        FILES_MATCHING PATTERN "*.track")
endif()
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)

