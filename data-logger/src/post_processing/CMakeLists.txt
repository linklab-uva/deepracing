add_library(${PROJECT_NAME}_post_processing
post_processing_utils.cpp
)
target_include_directories(${PROJECT_NAME}_post_processing PRIVATE ${ALGLIB_DIR}/..)
target_link_libraries(${PROJECT_NAME}_post_processing 
${PROJECT_NAME}_proto 
${PROJECT_NAME}_alglib 
${Boost_LIBRARIES}
)

add_executable(${PROJECT_NAME}_assign_labels
assign_labels.cpp
)
target_link_libraries(${PROJECT_NAME}_assign_labels 
${PROJECT_NAME}_post_processing
${Boost_LIBRARIES}
${YAML_CPP_LIBRARIES}
)


set(install_targets
#libraries
${PROJECT_NAME}_post_processing 
#executables
${PROJECT_NAME}_assign_labels
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)