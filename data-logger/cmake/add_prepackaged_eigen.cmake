message(STATUS "Building an up-to-date eigen library.")
set(eigen3_zip_ ${CMAKE_SOURCE_DIR}/third_party/eigen3.3.9.zip)
set(eigen3_working_dir_ ${CMAKE_BINARY_DIR}/Eigen3.3.9)
file(MAKE_DIRECTORY ${eigen3_working_dir_})
execute_process(
COMMAND ${CMAKE_COMMAND} -E tar xzf ${eigen3_zip_}
WORKING_DIRECTORY ${eigen3_working_dir_}
#DEPENDS ${eigen3_zip_}
#COMMENT "Unpacking Eigen3 zip file."
)
file(MAKE_DIRECTORY ${eigen3_working_dir_}/build)
# The first external project will be built at *configure stage*
execute_process(
    COMMAND ${CMAKE_COMMAND} ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${eigen3_working_dir_}/install
    WORKING_DIRECTORY ${eigen3_working_dir_}/build
    OUTPUT_QUIET
)
execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --target install
    WORKING_DIRECTORY ${eigen3_working_dir_}/build
    OUTPUT_QUIET
)
set(Eigen3_DIR ${eigen3_working_dir_}/install/share/eigen3/cmake PARENT_SCOPE)# "Path to Eigen3 installation directory.")
message("Built an up-to-date eigen version 3.3.90 at ${eigen3_working_dir_}/install.")