cmake_minimum_required(VERSION 3.15)
project(alglib_vendor CXX)

find_package(ament_cmake)
find_package(ament_cmake_vendor_package)

execute_process(COMMAND wget https://www.alglib.net/translator/re/alglib-4.04.0.cpp.gpl.zip)
file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/alglib-4.04.0.cpp.gpl.zip DESTINATION ${CMAKE_BINARY_DIR})

configure_file(alglibCMakeLists.txt.in ${CMAKE_BINARY_DIR}/alglib-cpp/CMakeLists.txt @ONLY)
configure_file(alglib-config.cmake.in ${CMAKE_BINARY_DIR}/alglib-cpp/alglib-config.cmake.in @ONLY)
include(ExternalProject)
# ament_vendor("alglibcpp" VCS_TYPE "path" VCS_URL "alglib-cpp")
ExternalProject_Add(alglibcpp
    SOURCE_DIR ${CMAKE_BINARY_DIR}/alglib-cpp
    LOG_CONFIGURE True
    LOG_BUILD True
    CMAKE_ARGS
        # -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/alglib_install
        -Wno-dev
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/alglib_install/
  DESTINATION opt/alglib
)

ament_environment_hooks(env_hook/alglib_vendor_library_path.sh)
if(APPLE)
  set(ENV_VAR_NAME "DYLD_LIBRARY_PATH")
else()
  set(ENV_VAR_NAME "LD_LIBRARY_PATH")
endif()
set(ENV_VAR_VALUE "opt/alglib/lib")
ament_environment_hooks(env_hook/alglib_vendor_library_path.dsv.in)
ament_package(CONFIG_EXTRAS cmake/alglib_extras.cmake.in)
