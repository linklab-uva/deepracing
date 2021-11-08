find_package(WIL REQUIRED)

add_executable(winrtcapture 
	main.cpp
	App.cpp
	SimpleCapture.cpp
)
target_link_libraries(winrtcapture PRIVATE WIL::WIL Dwmapi windowsapp)
target_link_options(winrtcapture PRIVATE "/manifestdependency:type=\'win32\' name=\'Microsoft.Windows.Common-Controls\' processorArchitecture=\'*\' language=\'*\' version=\'6.0.0.0\' publicKeyToken=\'6595b64144ccf1df\'")
set_target_properties(winrtcapture PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_graphics_capture_example WIN32_EXECUTABLE ON)
target_include_directories(winrtcapture
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(winrtcapture
	PRIVATE _UNICODE UNICODE _AFXDLL _BIND_TO_CURRENT_CRT_VERSION _BIND_TO_CURRENT_MFC_VERSION
)

install(TARGETS winrtcapture
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)