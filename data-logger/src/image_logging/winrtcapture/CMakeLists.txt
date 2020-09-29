add_library(winrt_capture SHARED
	WinrtCapture.cpp
)
target_include_directories(winrt_capture
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      #  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}/winrtcapture>
)
target_precompile_headers(winrt_capture
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
	"Win32WindowEnumeration.h"
)
target_link_libraries(winrt_capture
PUBLIC
	Dwmapi
	windowsapp
PRIVATE
	${OpenCV_LIBRARIES}
)
target_compile_definitions(winrt_capture
	PRIVATE F1_DATALOGGER_IMAGE_LOGGING_BUILDING_DLL
	INTERFACE WIN32_LEAN_AND_MEAN NOMINMAX
)
set_target_properties(winrt_capture PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_winrtcapture)

install(TARGETS winrt_capture
	EXPORT ${PROJECT_NAME}-export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES 
		"WinrtCapture.h"
		"Win32WindowEnumeration.h"
		"composition.interop.h"
		"d3dHelpers.h"
		"direct3d11.interop.h"
		"capture.interop.h"
	DESTINATION include/${PROJECT_NAME}/winrtcapture
)