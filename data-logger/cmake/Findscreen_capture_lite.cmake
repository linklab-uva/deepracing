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
	set(screen_capture_lite_LIBRARY_NAME "screen_capture_lite")
elseif(UNIX)
	list(APPEND LIBRARY_SEARCH_PATHS "/usr/lib")
	list(APPEND LIBRARY_SEARCH_PATHS "/usr/local/lib")
	set(screen_capture_lite_LIBRARY_NAME "screen_capture_lite")
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
