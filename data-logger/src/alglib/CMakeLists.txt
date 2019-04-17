set(_alglib_dir ${CMAKE_CURRENT_SOURCE_DIR} )
add_library(${PROJECT_NAME}_alglib
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
target_include_directories(${PROJECT_NAME}_alglib PRIVATE ${CMAKE_SOURCE_DIR}/include/f1_datalogger)

set(install_targets
#libraries.
${PROJECT_NAME}_alglib
)
install(TARGETS ${install_targets}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
unset(install_targets)
set(ALGLIB_DIR ${_alglib_dir} PARENT_SCOPE )
unset(_alglib_dir)
