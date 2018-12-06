include(FindProtobuf)
find_package(Protobuf REQUIRED)
protobuf_generate_cpp(${PROJECT_NAME}_PROTO_SRC ${PROJECT_NAME}_PROTO_HEADER F1UDPData.proto TimestampedImage.proto)
message("Generated sources: ${${PROJECT_NAME}_PROTO_SRC}")
message("Generated header: ${${PROJECT_NAME}_PROTO_HEADER}")
set(ProtobufIncludePath ${CMAKE_CURRENT_BINARY_DIR}
    CACHE INTERNAL "Path to generated protobuf files.")


add_library(${PROJECT_NAME}_proto ${${PROJECT_NAME}_PROTO_SRC})
target_link_libraries(${PROJECT_NAME}_proto ${Protobuf_LIBRARIES})
target_include_directories(${PROJECT_NAME}_proto PRIVATE ${ProtobufIncludePath})

add_executable(protobuf_test protobuf_test.cpp)
target_include_directories(protobuf_test PRIVATE ${ProtobufIncludePath} ${PROTOBUF_INCLUDE_DIR})
target_link_libraries(protobuf_test ${PROJECT_NAME}_proto)

