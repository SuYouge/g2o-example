# include_directories(${CSPARSE_INCLUDE_DIR})
# qt components used (also by qglviewer): Core Gui Xml OpenGL Widgets
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
find_package(QGLViewer REQUIRED)

include_directories(${CSPARSE_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

add_library(tutorial_slam2d_library ${G2O_LIB_TYPE}
  edge_se2_pointxy.cpp  se2.h                vertex_point_xy.h
  edge_se2.cpp    edge_se2_pointxy.h    vertex_se2.cpp
  edge_se2.h      vertex_point_xy.cpp  vertex_se2.h
  parameter_se2_offset.h parameter_se2_offset.cpp
  types_tutorial_slam2d.h types_tutorial_slam2d.cpp
  simulator.h simulator.cpp
  g2o_tutorial_slam2d_api.h
)

# set_target_properties(tutorial_slam2d_library PROPERTIES OUTPUT_NAME ${LIB_PREFIX}tutorial_slam2d)

target_link_libraries(tutorial_slam2d_library g2o_core g2o_solver_csparse g2o_csparse_extension cxsparse g2o_stuff)

add_executable(tutorial_slam2d
  tutorial_slam2d.cpp
)

# set_target_properties(tutorial_slam2d PROPERTIES OUTPUT_NAME tutorial_slam2d${EXE_POSTFIX})

target_link_libraries(tutorial_slam2d tutorial_slam2d_library)
