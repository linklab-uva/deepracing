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
