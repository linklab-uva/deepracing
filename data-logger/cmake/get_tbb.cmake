include(${THIRD_PARTY_REPOS}/tbb/cmake/TBBGet.cmake)
tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
find_package(TBB REQUIRED tbb)
message(STATUS "Unpacked TBB to : ${tbb_root}")
message(STATUS "TBB Configuration at : ${TBB_DIR}")
install(DIRECTORY ${TBB_DIR}/../bin/
    DESTINATION bin
)