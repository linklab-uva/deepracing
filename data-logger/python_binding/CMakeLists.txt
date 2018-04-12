cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_subdirectory(pybind11)
add_library(pyf1_datalogger MODULE pybind_module.cpp)
target_link_libraries(pyf1_datalogger PRIVATE ${PROJECT_NAME}_data_capture pybind11::module)
set_target_properties(pyf1_datalogger PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
