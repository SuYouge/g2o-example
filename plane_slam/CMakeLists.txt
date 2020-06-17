cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake_modules )

find_package(OpenCV 3 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED )
find_package(CSparse REQUIRED)
find_package(Cholmod REQUIRED)

include_directories(${CSPARSE_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

add_executable(simulator_3d_plane
  simulator_3d_plane.cpp
)
target_link_libraries(simulator_3d_plane g2o_core g2o_solver_csparse g2o_types_slam3d g2o_types_slam3d_addons g2o_stuff g2o_csparse_extension cxsparse)
# set_target_properties(simulator_3d_plane PROPERTIES OUTPUT_NAME simulator_3d_plane${EXE_POSTFIX})
