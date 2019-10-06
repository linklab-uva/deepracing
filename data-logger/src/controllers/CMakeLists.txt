if(MSVC)
    find_package(vJoy++ REQUIRED)
	set(controllers_SRCS
		pure_pursuit_controller.cpp
		windows/vjoy_interface.cpp
		windows/f1_interface_factory.cpp
	)
	set(additional_controller_libs_ vjoy::vJoy++)
	get_target_property(vjoy_library_type_ vjoy::vJoy++ TYPE)
	if("${vjoy_library_type_}" STREQUAL "SHARED_LIBRARY")
		string(TOUPPER ${CMAKE_BUILD_TYPE} dll_postfix_)
		get_target_property(vjoyppdll_ vjoy::vJoy++ IMPORTED_LOCATION_${dll_postfix_})
		add_custom_target(copy_vjoy_dll ALL
		COMMAND ${CMAKE_COMMAND} -E copy ${vjoyppdll_} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		DEPENDS ${PROJECT_NAME}_controllers
		)
	endif()
elseif(APPLE)
	set(controllers_SRCS
		pure_pursuit_controller.cpp
		ios/f1_interface_factory.cpp
	)
elseif(UNIX)
	set(controllers_SRCS
		pure_pursuit_controller.cpp
		linux/f1_interface_factory.cpp
	)
else()
	message(FATAL_ERROR "This library is not supported on your platform: ${CMAKE_SYSTEM}")
endif()
add_library(${PROJECT_NAME}_controllers ${controllers_SRCS})
if(UNIX)
	set_property(TARGET ${PROJECT_NAME}_controllers PROPERTY POSITION_INDEPENDENT_CODE ON)
endif(UNIX)
target_link_libraries(${PROJECT_NAME}_controllers
${PROJECT_NAME}_eigen_utils
${Boost_LIBRARIES}
Eigen3::Eigen
${additional_controller_libs_}
)
target_include_directories(${PROJECT_NAME}_controllers PRIVATE ${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR} )


add_dependencies(${PROJECT_NAME}_controllers ${PROJECT_NAME}_proto)
set_target_properties(${PROJECT_NAME}_controllers
					  PROPERTIES FOLDER vjoy)
					  
add_subdirectory(python_cpp_bindings)

set(install_targets
#executables.

#libraries
${PROJECT_NAME}_controllers
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)