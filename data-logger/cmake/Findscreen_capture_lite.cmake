# - Try to find screen_capture_lite
# Once done, this will define
#
#  screen_capture_lite_FOUND - system has screen_capture_lite
#  screen_capture_lite_INCLUDE_DIRS - the screen_capture_lite include directories
#  screen_capture_lite_LIBRARIES - link these to use screen_capture_lite

include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(screen_capture_lite_PKGCONF screen_capture_lite)

set(INCLUDE_SEARCH_PATHS "")
if(DEFINED ENV{SCREEN_CAPTURE_LITE_ROOT})
	list(APPEND INCLUDE_SEARCH_PATHS "$ENV{SCREEN_CAPTURE_LITE_ROOT}/include")
endif()
if(MSVC)
	list(APPEND INCLUDE_SEARCH_PATHS "C:/Program\ Files/screen_capture_lite/include")
	list(APPEND INCLUDE_SEARCH_PATHS "C:/Program\ Files\ (x86)/screen_capture_lite/include")
elseif(UNIX)
	list(APPEND INCLUDE_SEARCH_PATHS "/usr/include")
	list(APPEND INCLUDE_SEARCH_PATHS "/usr/local/include")
else()

endif()

set(screen_capture_lite_HEADER_FILE "ScreenCapture.h")


# Include dir
message("Looking for ${screen_capture_lite_HEADER_FILE} in ${INCLUDE_SEARCH_PATHS}")
find_path(screen_capture_lite_INCLUDE_DIR
  NAMES ${screen_capture_lite_HEADER_FILE}
  PATHS ${INCLUDE_SEARCH_PATHS}
)

set(LIBRARY_SEARCH_PATHS ${screen_capture_lite_PKGCONF_LIBRARY_DIRS})
if(DEFINED ENV{SCREEN_CAPTURE_LITE_ROOT})
	list(APPEND LIBRARY_SEARCH_PATHS "$ENV{SCREEN_CAPTURE_LITE_ROOT}/lib")
endif()
if(MSVC)
	list(APPEND LIBRARY_SEARCH_PATHS "C:/Program\ Files/screen_capture_lite/lib")
	list(APPEND LIBRARY_SEARCH_PATHS "C:/Program\ Files\ (x86)/screen_capture_lite/lib")
	set(screen_capture_lite_LIBRARY_NAME "screen_capture_lite.lib")
elseif(UNIX)
	list(APPEND LIBRARY_SEARCH_PATHS "/usr/lib")
	list(APPEND LIBRARY_SEARCH_PATHS "/usr/local/lib")
	if(${BUILD_SHARED_LIBS})
		set(screen_capture_lite_LIBRARY_NAME "libscreen_capture_lite.so")
	else()
		set(screen_capture_lite_LIBRARY_NAME "libscreen_capture_lite.a")
	endif()
else()
endif()

message("Looking for ${screen_capture_lite_LIBRARY_NAME} in ${LIBRARY_SEARCH_PATHS}")
# Finally the library itself
find_library(screen_capture_lite_LIBRARY
  NAMES ${screen_capture_lite_LIBRARY_NAME}
  PATHS ${LIBRARY_SEARCH_PATHS} 
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(screen_capture_lite_PROCESS_INCLUDES screen_capture_lite_INCLUDE_DIR)
set(screen_capture_lite_PROCESS_LIBS screen_capture_lite_LIBRARY)

libfind_process(screen_capture_lite)
if(WIN32)
	set(screen_capture_lite_PLATFORM_LIBS Dwmapi)
	add_definitions(-DNOMINMAX)
elseif(APPLE)
    find_package(Threads REQUIRED)
    find_library(corefoundation_lib CoreFoundation REQUIRED)
    find_library(cocoa_lib Cocoa REQUIRED)
    find_library(coremedia_lib CoreMedia REQUIRED)
    find_library(avfoundation_lib AVFoundation REQUIRED)
    find_library(coregraphics_lib CoreGraphics REQUIRED)
    find_library(corevideo_lib CoreVideo REQUIRED)
   
	set(screen_capture_lite_PLATFORM_LIBS
        ${CMAKE_THREAD_LIBS_INIT}
        ${corefoundation_lib}
        ${cocoa_lib}
        ${coremedia_lib}
        ${avfoundation_lib}
        ${coregraphics_lib}  
        ${corevideo_lib}
    ) 
else()
	find_package(X11 REQUIRED)
	if(!X11_XTest_FOUND)
 		message(FATAL_ERROR "X11 extensions are required, but not found!")
	endif()
	if(!X11_Xfixes_LIB)
 		message(FATAL_ERROR "X11 fixes extension is required, but not found!")
	endif()
	find_package(Threads REQUIRED)
	set(screen_capture_lite_PLATFORM_LIBS
		${X11_LIBRARIES}
		${X11_Xfixes_LIB}
		${X11_XTest_LIB}
		${X11_Xinerama_LIB}
		${CMAKE_THREAD_LIBS_INIT}
	)
#	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
endif()
