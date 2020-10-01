find_package(WIL REQUIRED)
add_library(winrt_capture_lib SHARED
	SimpleCapture.cpp
	CaptureSnapshot.cpp
	SimpleImageEncoder.cpp
	SampleWindow.cpp
	MonitorList.cpp
	WindowList.cpp
	App.cpp
)
target_include_directories(winrt_capture_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      #  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
		$<INSTALL_INTERFACE:include/${PROJECT_NAME}/winrtcapture>
		${Opencv_INCLUDE_DIRS}
)
target_precompile_headers(winrt_capture_lib
  PUBLIC

	<Unknwn.h>
	<inspectable.h>

	<wil/cppwinrt.h>

	# WinRT
	<winrt/Windows.Foundation.h>
	<winrt/Windows.Foundation.Collections.h>
	<winrt/Windows.Foundation.Metadata.h>
	<winrt/Windows.Storage.h>
	<winrt/Windows.Storage.Pickers.h>
	<winrt/Windows.Storage.Streams.h>
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

	# STL
	<atomic>
	<memory>
	<algorithm>
	<unordered_set>
	<vector>
	<optional>

	#D3D
	<d3d11_4.h>
	<dxgi1_6.h>
	<d2d1_3.h>
	<wincodec.h>

	#DWM
	<dwmapi.h>

	# WIL
	<wil/resource.h>

	#Helpers
	"util/composition.interop.h"
	"util/composition.desktop.interop.h"
	"util/d3dHelpers.h"
	"util/d3dHelpers.desktop.h"
	"util/direct3d11.interop.h"
	"util/capture.desktop.interop.h"
	"util/dispatcherqueue.desktop.interop.h"
	"util/stream.interop.h"
	"util/hwnd.interop.h"
	"completionSource.h"
)
target_compile_options(winrt_capture_lib PUBLIC /Zc:inline /fp:precise /await /Zc:forScope )# /Zc:forScope /GR 
target_link_libraries(winrt_capture_lib
PUBLIC
	Dwmapi
	windowsapp
	WIL::WIL
#PRIVATE
	${OpenCV_LIBRARIES}
)
target_compile_definitions(winrt_capture_lib
	PUBLIC _UNICODE UNICODE _RESUMABLE_ENABLE_LEGACY_AWAIT_ADAPTERS
	PRIVATE F1_DATALOGGER_IMAGE_LOGGING_BUILDING_DLL
	INTERFACE WIN32_LEAN_AND_MEAN NOMINMAX _AFXDLL _BIND_TO_CURRENT_CRT_VERSION _BIND_TO_CURRENT_MFC_VERSION
)
set_target_properties(winrt_capture_lib PROPERTIES 
OUTPUT_NAME ${PROJECT_NAME}_winrtcapture 
CXX_STANDARD 17
WINDOWS_EXPORT_ALL_SYMBOLS TRUE
#VS_WINRT_COMPONENT TRUE

)

add_executable(winrtcapture main.cpp)
target_link_libraries(winrtcapture winrt_capture_lib)
set_target_properties(winrtcapture PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_graphics_capture_example WIN32_EXECUTABLE ON)

install(TARGETS winrt_capture_lib winrtcapture
	EXPORT ${PROJECT_NAME}-export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
	DESTINATION include/${PROJECT_NAME}/winrtcapture
	FILES_MATCHING PATTERN "*.h"
)