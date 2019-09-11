include(FindPython3)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_subdirectory(pybind11)

#Python3::Module
#Python3::Python
pybind11_add_module(py_f1_interface py_f1_interface.cpp)
target_link_libraries(py_f1_interface
PRIVATE
Python3::Module
Python3::Python
pybind11::module
pybind11::pybind11
${PROJECT_NAME}_controllers
)
target_include_directories(py_f1_interface
PRIVATE
${PYBIND11_INCLUDE_DIR}
)
target_compile_definitions(py_f1_interface PRIVATE NOMINMAX=1)
target_compile_definitions(py_f1_interface PRIVATE VERSION_INFO="1.0.0")
set_target_properties(py_f1_interface PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
set(install_targets
#executables.

#libraries
py_f1_interface
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib/python
    ARCHIVE DESTINATION lib
)
unset(install_targets)
