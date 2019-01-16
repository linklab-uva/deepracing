add_executable(${PROJECT_NAME}_assign_labels
assign_labels.cpp
)
target_include_directories(${PROJECT_NAME}_assign_labels PRIVATE ${ALGLIB_DIR}/..)
target_link_libraries(${PROJECT_NAME}_assign_labels 
${PROJECT_NAME}_proto 
${PROJECT_NAME}_alglib 
${Boost_LIBRARIES}
)
