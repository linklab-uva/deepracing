
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)


# Populate a CMake variable with the sources
set(vjoycalibration_SRCS
    vjoycalibration/vjoycalibrationwindow.ui
    vjoycalibration/vjoycalibrationwindow.cpp
    vjoycalibration/main.cpp
)
# Tell CMake to create the helloworld executable
add_executable(vjoycalibrationgui WIN32 ${vjoycalibration_SRCS})
# Use the Widgets module from Qt 5
target_link_libraries(vjoycalibrationgui Qt5::Widgets)
