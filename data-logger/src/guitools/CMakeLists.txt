
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
# Tell CMake to create the vjoycalibrationgui executable
add_library(${PROJECT_NAME}_datacapturewindow ${datacapture_SRCS})
target_link_libraries(${PROJECT_NAME}_datacapturewindow Qt5::Widgets ${YAML_CPP_LIBRARIES})

if(MSVC)
    add_executable(${PROJECT_NAME}_datacapturegui WIN32 datacapture/main.cpp)
else()
    add_executable(${PROJECT_NAME}_datacapturegui datacapture/main.cpp)
endif()

target_link_libraries(${PROJECT_NAME}_datacapturegui ${PROJECT_NAME}_datacapturewindow)
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