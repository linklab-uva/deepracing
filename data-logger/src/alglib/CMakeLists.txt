set(${PROJECT_NAME}_math_utils_src
alglibinternal.cpp
alglibmisc.cpp
ap.cpp
dataanalysis.cpp
diffequations.cpp
fasttransforms.cpp
integration.cpp
interpolation.cpp
linalg.cpp
optimization.cpp
solvers.cpp
specialfunctions.cpp
statistics.cpp
)
add_library(${PROJECT_NAME}_math_utils ${${PROJECT_NAME}_math_utils_src})
target_include_directories(${PROJECT_NAME}_math_utils PRIVATE ${CMAKE_SOURCE_DIR}/include/f1_datalogger)

set(install_targets
#libraries.
${PROJECT_NAME}_math_utils
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)
