if(MSVC)
    find_package(vJoy_plusplus CONFIG REQUIRED)
	set_target_properties(vjoy::vJoy_plusplus PROPERTIES IMPORTED_GLOBAL TRUE)
	set(controllers_SRCS
		windows/vjoy_interface.cpp
		windows/f1_interface_factory.cpp
	)
	set(additional_controller_libs_ vjoy::vJoy_plusplus)
	set(USING_VJOY TRUE)
	set(USING_VJOY TRUE PARENT_SCOPE)

	find_package(ViGEmClient CONFIG QUIET)
	if(TARGET vigem_client::vigem_client)
		message(STATUS "Found ViGEmClient")
		list(APPEND controllers_SRCS windows/multiagent_f1_interface_factory.cpp)
		list(APPEND controllers_SRCS windows/vigem_interface.cpp)
		list(APPEND additional_controller_libs_ vigem_client::vigem_client)
		list(APPEND additional_controller_libs_ setupAPI)
		set(USING_VIGEM TRUE)
		set(USING_VIGEM TRUE PARENT_SCOPE)
	endif()
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

message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "Building controllers library with: ${controllers_SRCS}")
add_library(${PROJECT_NAME}_controllers ${controllers_SRCS})
if(UNIX)
	set_target_properties(${PROJECT_NAME}_controllers PROPERTIES POSITION_INDEPENDENT_CODE ON)
elseif(MSVC)
	target_compile_definitions(${PROJECT_NAME}_controllers PRIVATE "WIN32_LEAN_AND_MEAN" "NOGDI")
endif()
set_target_properties(${PROJECT_NAME}_controllers PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED TRUE)

target_link_libraries(${PROJECT_NAME}_controllers
PUBLIC
	${PROJECT_NAME}_eigen_utils
	${PROJECT_NAME}_udp_utils
	${PROJECT_NAME}_udp_logging_common
	${Boost_TARGETS}
	Eigen3::Eigen
PRIVATE
	${additional_controller_libs_}
)
target_include_directories(${PROJECT_NAME}_controllers PRIVATE ${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR} )
set_target_properties(${PROJECT_NAME}_controllers PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(${PROJECT_NAME}_controllers PRIVATE "F1_DATALOGGER_CONTROLS_BUILDING_DLL")
add_dependencies(${PROJECT_NAME}_controllers ${PROJECT_NAME}_proto)
					  
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