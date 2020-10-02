find_package(WIL REQUIRED)
add_library(winrt_capture_pch INTERFACE)
target_precompile_headers(winrt_capture_pch
INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pch.h>
	$<INSTALL_INTERFACE:\${_IMPORT_PREFIX}/include/${PROJECT_NAME}/image_logging/winrtcapture/pch.h>
)
install(FILES pch.h DESTINATION include/${PROJECT_NAME}/image_logging/winrtcapture)
add_library(winrt_capture_lib SHARED
	lib/SimpleCapture.cpp
	lib/MonitorList.cpp
	lib/WindowList.cpp
)
target_include_directories(winrt_capture_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
      #  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
		#$<INSTALL_INTERFACE:include/${PROJECT_NAME}/winrtcapture>
		${Opencv_INCLUDE_DIRS}
)

target_compile_options(winrt_capture_lib PUBLIC /Zc:inline /fp:precise /await /Zc:forScope )# /Zc:forScope /GR 
target_link_libraries(winrt_capture_lib
PUBLIC
	Dwmapi
	windowsapp
	WIL::WIL
	${OpenCV_LIBRARIES}
PRIVATE
	winrt_capture_pch
)
target_compile_definitions(winrt_capture_lib
	PUBLIC _UNICODE UNICODE 
	PRIVATE F1_DATALOGGER_IMAGE_LOGGING_BUILDING_DLL 
	INTERFACE WIN32_LEAN_AND_MEAN NOMINMAX _AFXDLL _BIND_TO_CURRENT_CRT_VERSION _BIND_TO_CURRENT_MFC_VERSION
)
set_target_properties(winrt_capture_lib PROPERTIES 
OUTPUT_NAME ${PROJECT_NAME}_winrtcapture 
CXX_STANDARD 17
WINDOWS_EXPORT_ALL_SYMBOLS TRUE
#VS_WINRT_COMPONENT TRUE

)

add_executable(winrtcapture 
exampleapp/CaptureSnapshot.cpp
exampleapp/SimpleImageEncoder.cpp
exampleapp/SampleWindow.cpp
exampleapp/App.cpp
exampleapp/main.cpp
)
target_link_libraries(winrtcapture PRIVATE winrt_capture_lib winrt_capture_pch)
target_link_options(winrtcapture PRIVATE "/manifestdependency:type=\'win32\' name=\'Microsoft.Windows.Common-Controls\' processorArchitecture=\'*\' language=\'*\' version=\'6.0.0.0\' publicKeyToken=\'6595b64144ccf1df\'")
set_target_properties(winrtcapture PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_graphics_capture_example WIN32_EXECUTABLE ON)
target_include_directories(winrt_capture_lib
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/exampleapp ${CMAKE_CURRENT_SOURCE_DIR}
)

install(TARGETS winrt_capture_lib winrt_capture_pch winrtcapture
	EXPORT ${PROJECT_NAME}-export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)