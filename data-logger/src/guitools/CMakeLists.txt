
find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5Core CONFIG REQUIRED)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)


# Populate a CMake variable with the sources

set(datacapture_SRCS
    datacapture/datacapture.ui
    datacapture/datacapture.cpp
)
#set(QT_USE_QTMAIN FALSE)
# Tell CMake to create the vjoycalibrationgui executable
add_library(${PROJECT_NAME}_datacapturewindow ${datacapture_SRCS})
target_link_libraries(${PROJECT_NAME}_datacapturewindow Qt5::Widgets ${YAML_CPP_LIBRARIES})

if(MSVC)
    add_executable(${PROJECT_NAME}_datacapturegui WIN32 datacapture/main.cpp)
	set(extraQTLIBS Qt5::WinMain)
	set_target_properties(${PROJECT_NAME}_datacapturegui ${PROJECT_NAME}_datacapturewindow PROPERTIES Qt5_NO_LINK_QTMAIN OFF)
	get_target_property(_qmake_executable Qt5::qmake IMPORTED_LOCATION)
	get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)
	find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${_qt_bin_dir}")
	add_custom_target(deployqt ALL
    COMMAND "${CMAKE_COMMAND}" -E
        env PATH="${_qt_bin_dir}" "${WINDEPLOYQT_EXECUTABLE}"
            --verbose 0
            --no-compiler-runtime
            \"$<TARGET_FILE:${PROJECT_NAME}_datacapturegui>\"
    COMMENT "Deploying Qt..."
	)
	add_dependencies(deployqt ${PROJECT_NAME}_datacapturegui)
else()
    add_executable(${PROJECT_NAME}_datacapturegui datacapture/main.cpp)
endif()
target_link_libraries(${PROJECT_NAME}_datacapturegui ${PROJECT_NAME}_datacapturewindow ${extraQTLIBS})
set_target_properties(${PROJECT_NAME}_datacapturegui ${PROJECT_NAME}_datacapturewindow
					  PROPERTIES FOLDER guitools)
set(install_targets
#executables.
${PROJECT_NAME}_datacapturegui
#libraries
${PROJECT_NAME}_datacapturewindow
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)