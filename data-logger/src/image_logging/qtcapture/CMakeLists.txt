cmake_minimum_required(VERSION 3.14)


set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Core Widgets)

add_executable(consolegrabber
  main.cpp
)
target_compile_definitions(consolegrabber
	PUBLIC _UNICODE UNICODE WIN32_LEAN_AND_MEAN NOMINMAX _AFXDLL _BIND_TO_CURRENT_CRT_VERSION _BIND_TO_CURRENT_MFC_VERSION
)
target_include_directories(consolegrabber PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OPENCV_INCLUDE_DIRS})
target_link_libraries(consolegrabber Qt5::Core Qt5::Widgets dwmapi windowsapp ${OpenCV_LIBRARIES})
set_target_properties(consolegrabber PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_consolegrabber)# WIN32_EXECUTABLE ON)


install(TARGETS consolegrabber
	EXPORT ${PROJECT_NAME}-export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)