#include(FindArmadillo)
add_executable(${PROJECT_NAME}_arma_test arma_test.cpp)
target_include_directories(${PROJECT_NAME}_arma_test PRIVATE ${ARMADILLO_INCLUDE_DIRS} ${ProtobufIncludePath} ${Protobuf_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}_arma_test ${ARMADILLO_LIBRARIES} Eigen3::Eigen ${PROJECT_NAME}_eigen_utils ${PROJECT_NAME}_proto)