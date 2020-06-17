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

if(CHOLMOD_INCLUDE_DIR OR CSPARSE_INCLUDE_DIR)
  add_executable(bal_example bal_example.cpp)
endif()

# Link CHOLMOD if available, CSparse otherwise
if(CHOLMOD_FOUND)
  include_directories(${CHOLMOD_INCLUDE_DIR})
  target_link_libraries(bal_example g2o_solver_cholmod)
elseif(CSPARSE_FOUND)
  include_directories(${CSPARSE_INCLUDE_DIR})
  target_link_libraries(bal_example g2o_solver_csparse)
endif()
