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
  add_executable(sba_demo sba_demo.cpp)
  target_link_libraries(sba_demo g2o_core g2o_stuff g2o_types_icp g2o_types_sba)
  # set_target_properties(sba_demo PROPERTIES OUTPUT_NAME sba_demo${EXE_POSTFIX})
endif()

# Link CHOLMOD if available, CSparse otherwise
if(CHOLMOD_FOUND)
  include_directories(${CHOLMOD_INCLUDE_DIR})
  target_link_libraries(sba_demo g2o_core g2o_stuff g2o_solver_cholmod cholmod g2o_types_slam3d)
elseif(CSPARSE_FOUND)
  include_directories(${CSPARSE_INCLUDE_DIR})
  target_link_libraries(sba_demo g2o_core g2o_stuff g2o_solver_csparse g2o_csparse_extension cxsparse g2o_types_slam3d)
endif()
