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

add_executable(static_target
  static_target.cpp
)
target_link_libraries(static_target g2o_core g2o_stuff g2o_solver_cholmod cholmod)
# set_target_properties(static_target PROPERTIES OUTPUT_NAME static_target${EXE_POSTFIX})

add_executable(constant_velocity_target
  constant_velocity_target.cpp
)

# set_target_properties(constant_velocity_target PROPERTIES OUTPUT_NAME constant_velocity_target${EXE_POSTFIX})

target_link_libraries(constant_velocity_target g2o_core g2o_stuff g2o_solver_cholmod g2o_solver_pcg g2o_types_slam2d cholmod)


