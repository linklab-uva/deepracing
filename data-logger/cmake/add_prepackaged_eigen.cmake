set(eigen3_zip_ ${CMAKE_SOURCE_DIR}/third_party/eigen_3_3_90.zip)
execute_process(
COMMAND ${CMAKE_COMMAND} -E tar xzf ${eigen3_zip_}
WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#DEPENDS ${eigen3_zip_}
#COMMENT "Unpacking Eigen3 zip file."
)
set(eigen3_thirdparty_dir ${CMAKE_BINARY_DIR}/eigen_3_3_90)
set(Eigen3_DIR ${eigen3_thirdparty_dir}/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)