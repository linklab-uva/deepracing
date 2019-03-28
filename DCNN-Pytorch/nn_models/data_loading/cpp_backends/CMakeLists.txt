project(deepf1_image_backends)
cmake_minimum_required(VERSION 3.12)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem regex)
add_subdirectory(pybind11)
#find_package(PythonLibs 3 REQUIRED)
# add_executable(imageReader imageReader.cpp)
# target_include_directories(imageReader PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
# target_link_libraries(imageReader ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
option(BUILD_SHARED_LIBS "build shared libraries" ON)

add_library(opencv_numpy
pybind11_opencv_numpy/ndarray_converter.cpp
)
target_include_directories(opencv_numpy PRIVATE pybind11_opencv_numpy ${OpenCV_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(opencv_numpy ${PYTHON_LIBRARIES} ${OpenCV_LIBS})

add_library(deepf1_image_reading MODULE cv_image_reading.cpp)
target_link_libraries(deepf1_image_reading PRIVATE pybind11::module opencv_numpy ${PYTHON_LIBRARIES})
target_include_directories(deepf1_image_reading PRIVATE pybind11_opencv_numpy)
set_target_properties(deepf1_image_reading PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")