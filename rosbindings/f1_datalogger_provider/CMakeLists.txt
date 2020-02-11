cmake_minimum_required(VERSION 3.12)
project(f1_datalogger_provider)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include(f1_datalogger_provider-extras.cmake)

get_filename_component(f1_cmake_directory_ "${f1_datalogger_DIR}" REALPATH)
message(STATUS "f1_cmake_directory_: ${f1_cmake_directory_}")
set(f1_python_directory_ "${f1_cmake_directory_}/../lib/python")
message(STATUS "f1_python_directory_: ${f1_python_directory_}")
get_filename_component(f1_python_directory_ "${f1_python_directory_}" REALPATH)
message(STATUS "f1_python_directory_ after resolving symlinks: ${f1_python_directory_}")

if(WIN32)
  configure_file(addpythonpath.bat.in addpythonpath.bat @ONLY)
  ament_environment_hooks(${CMAKE_BINARY_DIR}/addpythonpath.bat)
else()
  configure_file(addpythonpath.bash.in addpythonpath.bash @ONLY)
  ament_environment_hooks(${CMAKE_BINARY_DIR}/addpythonpath.bash)
endif(WIN32)

if(WIN32)
  configure_file(addpythonpath.bat.in addpythonpath.bat @ONLY)
  ament_environment_hooks(${CMAKE_BINARY_DIR}/addpythonpath.bat)
else()
  configure_file(addpythonpath.bash.in addpythonpath.bash @ONLY)
  ament_environment_hooks(${CMAKE_BINARY_DIR}/addpythonpath.bash)
endif(WIN32)

ament_package(
  CONFIG_EXTRAS "f1_datalogger_provider-extras.cmake"
)
