
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
add_library(datacapturewindow ${datacapture_SRCS})
target_link_libraries(datacapturewindow Qt5::Widgets)

if(MSVC)
    add_executable(datacapturegui WIN32 datacapture/main.cpp)
else()
    add_executable(datacapturegui datacapture/main.cpp)
endif()

target_link_libraries(datacapturegui datacapturewindow)
set_target_properties(datacapturegui datacapturewindow
					  PROPERTIES FOLDER guitools)