
find_package(Python REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 REQUIRED)

pybind11_add_module(py_f1_interface py_f1_interface.cpp)
target_link_libraries(py_f1_interface
PUBLIC
Python::Module
Python::Python
pybind11::module
pybind11::pybind11
${PROJECT_NAME}_controllers
)
if(UNIX)
	set_property(TARGET py_f1_interface PROPERTY POSITION_INDEPENDENT_CODE ON)
endif(UNIX)
target_include_directories(py_f1_interface
PRIVATE
${PYBIND11_INCLUDE_DIR}
)
target_compile_definitions(py_f1_interface PRIVATE NOMINMAX=1)
target_compile_definitions(py_f1_interface PRIVATE VERSION_INFO="1.0.0")
set_target_properties(py_f1_interface PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
#set_target_properties(py_f1_interface PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

set(install_targets
#executables.
#libraries
py_f1_interface
)
if(ament_cmake_FOUND)
    install(TARGETS ${install_targets}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib/site-packages
        ARCHIVE DESTINATION lib
    )
else()
    install(TARGETS ${install_targets}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib/python
        ARCHIVE DESTINATION lib
    )
endif()
unset(install_targets)