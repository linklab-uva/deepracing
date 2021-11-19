
add_library(${PROJECT_NAME}_winrtcapture 
SimpleCapture.cpp
)
target_include_directories(${PROJECT_NAME}_winrtcapture 
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/f1_datalogger/image_logging/winrt>
)
target_compile_options(${PROJECT_NAME}_winrtcapture  PUBLIC "/Zc:__cplusplus")
target_compile_definitions(${PROJECT_NAME}_winrtcapture 
	PUBLIC _UNICODE UNICODE _AFXDLL _BIND_TO_CURRENT_CRT_VERSION _BIND_TO_CURRENT_MFC_VERSION "_WIN32_WINNT=${win32_ver}"
    PRIVATE F1_DATALOGGER_WINRT_CAPTURE_BUILDING_DLL
)
target_link_libraries(${PROJECT_NAME}_winrtcapture 
    PUBLIC Dwmapi windowsapp opencv_core opencv_highgui
)

add_executable(winrtcapture_example
	maingui.cpp
	App.cpp
)
target_link_libraries(winrtcapture_example PRIVATE ${PROJECT_NAME}_winrtcapture)
set_target_properties(winrtcapture_example PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_graphics_capture_example WIN32_EXECUTABLE ON)
install(TARGETS winrtcapture_example
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES 
    common.h composition.interop.h d3dHelpers.h direct3d11.interop.h capture.interop.h Win32WindowEnumeration.h
    DESTINATION include/f1_datalogger/image_logging/winrt
)