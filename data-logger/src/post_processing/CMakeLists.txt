add_library(${PROJECT_NAME}_post_processing
post_processing_utils.cpp
post_processing_utils2018.cpp
)
#add_dependencies(${PROJECT_NAME}_post_processing protocopy)
target_include_directories(${PROJECT_NAME}_post_processing PRIVATE ${ALGLIB_DIR}/..)
target_link_libraries(${PROJECT_NAME}_post_processing 
${PROJECT_NAME}_proto 
${Boost_TARGETS}
)

add_executable(${PROJECT_NAME}_assign_labels
assign_labels.cpp
)
#add_dependencies(${PROJECT_NAME}_assign_labels protocopy)
target_link_libraries(${PROJECT_NAME}_assign_labels 
${PROJECT_NAME}_post_processing
${Boost_TARGETS}
${YAML_CPP_LIBRARIES}
${OpenCV_LIBRARIES}
)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_link_libraries(${PROJECT_NAME}_assign_labels stdc++fs )
endif()


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