if(MSVC)
    find_package(vJoy_plusplus CONFIG REQUIRED)
	set_target_properties(vjoy::vJoy_plusplus PROPERTIES IMPORTED_GLOBAL TRUE)
	set(controllers_SRCS
		windows/vjoy_interface.cpp
		windows/f1_interface_factory.cpp
	)
	set(additional_controller_libs_ vjoy::vJoy_plusplus)
elseif(APPLE)
	set(controllers_SRCS
		ios/f1_interface_factory.cpp
	)
elseif(UNIX)
	set(controllers_SRCS
		linux/f1_interface_factory.cpp
	)
else()
	message(FATAL_ERROR "This library is not supported on your platform: ${CMAKE_SYSTEM}")
endif()

if(${Eigen3_VERSION} VERSION_GREATER_EQUAL "3.3.9")
	list(APPEND controllers_SRCS pure_pursuit_controller.cpp)
endif()
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "Building controllers library with: ${controllers_SRCS}")
add_library(${PROJECT_NAME}_controllers ${controllers_SRCS})
if(UNIX)
	set_property(TARGET ${PROJECT_NAME}_controllers PROPERTY POSITION_INDEPENDENT_CODE ON)
endif(UNIX)
target_link_libraries(${PROJECT_NAME}_controllers
${PROJECT_NAME}_eigen_utils
${PROJECT_NAME}_udp_utils
${PROJECT_NAME}_udp_logging_common
${Boost_TARGETS}
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