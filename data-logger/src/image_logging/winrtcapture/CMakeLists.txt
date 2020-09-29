add_library(winrt_capture_lib SHARED
	SimpleCapture.cpp
	CaptureApp.cpp
)
target_include_directories(winrt_capture_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      #  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}/winrtcapture>
)
target_precompile_headers(winrt_capture_lib
PUBLIC
	<d3d12.h>
	<Unknwn.h>
	<inspectable.h>
	<winrt/Windows.Foundation.h>
	<winrt/Windows.System.h>
	<winrt/Windows.UI.h>
	<winrt/Windows.UI.Composition.h>
	<winrt/Windows.UI.Composition.Desktop.h>
	<winrt/Windows.UI.Popups.h>
	<winrt/Windows.Graphics.Capture.h>
	<winrt/Windows.Graphics.DirectX.h>
	<winrt/Windows.Graphics.DirectX.Direct3d11.h>
	<windows.ui.composition.interop.h>
	<DispatcherQueue.h>
	<atomic>
	<memory>
	<d3d11_4.h>
	<dxgi1_6.h>
	<d2d1_3.h>
	<wincodec.h>
    <codecvt>
	"composition.interop.h"
	"d3dHelpers.h"
	"direct3d11.interop.h"
	"capture.interop.h"
)
target_link_libraries(winrt_capture_lib
PUBLIC
	Dwmapi
	windowsapp
PRIVATE
	${OpenCV_LIBRARIES}
)
target_compile_definitions(winrt_capture_lib
	PUBLIC _UNICODE UNICODE
	PRIVATE F1_DATALOGGER_IMAGE_LOGGING_BUILDING_DLL
	INTERFACE WIN32_LEAN_AND_MEAN NOMINMAX _AFXDLL _BIND_TO_CURRENT_CRT_VERSION _BIND_TO_CURRENT_MFC_VERSION
)
set_target_properties(winrt_capture_lib PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_winrtcapture)

add_executable(winrtcapture main.cpp)
target_link_libraries(winrtcapture winrt_capture_lib)
set_target_properties(winrtcapture PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_graphics_capture_example WIN32_EXECUTABLE ON)

install(TARGETS winrt_capture_lib winrtcapture
	EXPORT ${PROJECT_NAME}-export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES 
		"CaptureApp.h"
		"SimpleCapture.h"
		"Win32WindowEnumeration.h"
		"composition.interop.h"
		"d3dHelpers.h"
		"direct3d11.interop.h"
		"capture.interop.h"
	DESTINATION include/${PROJECT_NAME}/winrtcapture
)