cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake_modules )

# find_package(OpenCV 3 REQUIRED)
find_package(G2O REQUIRED)
# find_package(Sophus REQUIRED)
find_package( Eigen3 REQUIRED )
find_package( CSparse REQUIRED )
find_package( SuiteSparse REQUIRED )
find_package( Cholmod REQUIRED )

include_directories(
        # ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        # ${Sophus_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CSPARSE_INCLUDE_DIR}
        ${SUITESPARSE_INCLUDE_DIR}
        ${CHOLMOD_INCLUDE_DIR}
)

if(CSPARSE_INCLUDE_DIR)
  add_executable(circle_fit circle_fit.cpp)
  target_link_libraries(
    circle_fit g2o_core 
    g2o_solver_csparse 
    g2o_stuff 
    g2o_csparse_extension 
    cxsparse
    )
endif()

add_executable(curve_fit curve_fit.cpp)
target_link_libraries(curve_fit g2o_core g2o_stuff )
