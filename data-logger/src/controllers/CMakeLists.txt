if(MSVC)
    find_package(vJoy++ REQUIRED)
	set(controllers_SRCS
		pure_pursuit_controller.cpp
		windows/vjoy_interface.cpp
		windows/f1_interface_factory.cpp
	)
	set(additional_controller_libs_ ${vJoy++_LIBRARIES})
	set(additional_controller_includes_ ${vJoy++_INCLUDE_DIR})
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
target_link_libraries(${PROJECT_NAME}_controllers
${Boost_LIBRARIES}
Eigen3::Eigen
${additional_controller_libs_}
)
target_include_directories(${PROJECT_NAME}_controllers PRIVATE ${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR} ${additional_controller_includes_})

add_dependencies(${PROJECT_NAME}_controllers ${PROJECT_NAME}_proto)
set_target_properties(${PROJECT_NAME}_controllers
                      PROPERTIES FOLDER vjoy)
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