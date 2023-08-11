message(STATUS "Building waymo dataset conversions")

if(NOT (EXISTS ${CMAKE_BINARY_DIR}/tensorflow_src/tensorflow-2.13.0))
    message(STATUS "Downloading tensorflow source code, this may take a while")
    file(DOWNLOAD https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.13.0.tar.gz tensorflow.tar.gz)
    message(STATUS "Extracting tensorflow source code, this may also take a while")
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/tensorflow.tar.gz DESTINATION ${CMAKE_BINARY_DIR}/tensorflow_src)
endif()

set(tensorflow_source_dir ${CMAKE_BINARY_DIR}/tensorflow_src/tensorflow-2.13.0)
set(tensorflow_framework_proto_dir ${tensorflow_source_dir}/tensorflow/core/framework)
set(tensorflow_example_proto_dir ${tensorflow_source_dir}/tensorflow/core/example)
set(tensorflow_tsl_proto_dir ${tensorflow_source_dir}/tensorflow/tsl/protobuf)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/tsl)
file(COPY ${tensorflow_tsl_proto_dir}
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/tsl/
    FILES_MATCHING PATTERN "*.proto" 
)
file(GLOB TF_TSL_PROTOFILES
  "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/tsl/protobuf/*.proto"
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/core)
file(COPY ${tensorflow_example_proto_dir}
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/core/
    FILES_MATCHING PATTERN "*.proto" 
)
file(GLOB TF_EXAMPLE_PROTOFILES
  "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/core/example/*.proto"
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/core)
file(COPY ${tensorflow_framework_proto_dir}
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/core/
    FILES_MATCHING PATTERN "*.proto"
)
file(GLOB TF_FRAMEWORK_PROTOFILES
  "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/core/framework/*.proto"
)

set(tf-proto-files
    ${TF_EXAMPLE_PROTOFILES}
    ${TF_FRAMEWORK_PROTOFILES}
    ${TF_TSL_PROTOFILES}
)

add_library(tf-proto-objects OBJECT ${tf-proto-files})
# target_link_libraries(tf-proto-objects PRIVATE protobuf::libprotobuf)
# target_link_libraries(tf-proto-objects PRIVATE tensorflow::tensorflow)
target_include_directories(tf-proto-objects PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>")
protobuf_generate(TARGET tf-proto-objects
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    PROTOC_OUT_DIR ${CMAKE_BINARY_DIR}
)

set(waymo_dataset_dir ${CMAKE_BINARY_DIR}/waymo_open_dataset_src/waymo-open-dataset-1.5.2)
if(NOT (EXISTS ${waymo_dataset_dir}))
    message(STATUS "Downloading waymo open dataset source code, this may take a while")
    file(DOWNLOAD https://github.com/waymo-research/waymo-open-dataset/archive/refs/tags/v1.5.2.tar.gz waymo_open_dataset.tar.gz)
    message(STATUS "Extracting waymo open dataset source code, this may also take a while")
    file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/waymo_open_dataset.tar.gz DESTINATION ${CMAKE_BINARY_DIR}/waymo_open_dataset_src)
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/waymo_open_dataset)
file(COPY ${waymo_dataset_dir}/src/waymo_open_dataset/protos
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/waymo_open_dataset/
    FILES_MATCHING PATTERN "*.proto" 
)
file(COPY ${waymo_dataset_dir}/src/waymo_open_dataset/dataset.proto DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/waymo_open_dataset/)
file(COPY ${waymo_dataset_dir}/src/waymo_open_dataset/label.proto DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/waymo_open_dataset/)


file(GLOB WAYMO_PROTOFILES
  "${CMAKE_CURRENT_SOURCE_DIR}/waymo_open_dataset/protos/*.proto"
)
set(waymo-proto-files
    waymo_open_dataset/dataset.proto
    waymo_open_dataset/label.proto
    ${WAYMO_PROTOFILES}
)

add_library(waymo-proto-objects OBJECT ${waymo-proto-files})
# target_link_libraries(waymo-proto-objects PRIVATE protobuf::libprotobuf)
# target_link_libraries(waymo-proto-objects PRIVATE tensorflow::tensorflow)
target_include_directories(waymo-proto-objects PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>")
protobuf_generate(TARGET waymo-proto-objects
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    PROTOC_OUT_DIR ${CMAKE_BINARY_DIR}
)

set_target_properties(waymo-proto-objects PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(tf-proto-objects PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(waymo_conversions SHARED ${waymo_dataset_dir}/src/waymo_open_dataset/data_conversion/scenario_conversion.cc)
target_link_libraries(waymo_conversions 
    PUBLIC 
        protobuf::libprotobuf
        glog::glog 
        absl::any
        tf-proto-objects
        waymo-proto-objects 
)
target_include_directories(waymo_conversions 
    PUBLIC 
        "$<BUILD_INTERFACE:${waymo_dataset_dir}/src>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>"
)


find_package(pybind11 REQUIRED)

pybind11_add_module(py_waymo_conversions pybind11_module.cpp)
target_link_libraries(py_waymo_conversions
    PUBLIC
        Python::Module
        Python::Python
        pybind11::module
        pybind11::pybind11
        waymo_conversions
)
if(UNIX)
	set_property(TARGET py_waymo_conversions PROPERTY POSITION_INDEPENDENT_CODE ON)
endif(UNIX)
target_include_directories(py_waymo_conversions
    PRIVATE
        ${PYBIND11_INCLUDE_DIR}
)
target_compile_definitions(py_waymo_conversions PRIVATE NOMINMAX=1)
target_compile_definitions(py_waymo_conversions PRIVATE VERSION_INFO="1.0.0")
set_target_properties(py_waymo_conversions PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")

message(STATUS "Python3_SITELIB: ${Python3_SITELIB}")
set(pymoduleinstall lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/deepracing)
install(TARGETS py_waymo_conversions
    RUNTIME DESTINATION ${pymoduleinstall}
    LIBRARY DESTINATION ${pymoduleinstall}
    ARCHIVE DESTINATION ${pymoduleinstall}
)
install(TARGETS waymo_conversions waymo-proto-objects tf-proto-objects 
    EXPORT ${PROJECT_NAME}_export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)