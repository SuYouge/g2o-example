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

# add_executable(orb_cv orb_cv.cpp)
# target_link_libraries(orb_cv ${OpenCV_LIBS})

# add_executable(orb_self orb_self.cpp)
# target_link_libraries(orb_self ${OpenCV_LIBS})

# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
# add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
# target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

# # add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2
# add_executable(triangulation triangulation.cpp)
# target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pnp_self_defined pnp_self_defined.cpp)
target_link_libraries(pnp_self_defined
        g2o_core g2o_stuff
        ${OpenCV_LIBS})

add_executable(icp_self_defined icp_self_defined.cpp)
target_link_libraries(icp_self_defined
        g2o_core g2o_stuff
        ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d
        g2o_core g2o_stuff g2o_solver_cholmod g2o_types_sba g2o_solver_csparse g2o_csparse_extension cholmod 
        ${OpenCV_LIBS})

add_executable(icp_g2o_defined icp_g2o_defined.cpp)
target_link_libraries(icp_g2o_defined
        g2o_core g2o_stuff g2o_types_sba g2o_types_slam3d g2o_types_icp g2o_solver_cholmod cholmod 
        ${OpenCV_LIBS})

        
add_executable(pnp_g2o_defined pnp_g2o_defined.cpp)
target_link_libraries(pnp_g2o_defined
        g2o_core g2o_stuff g2o_solver_cholmod cholmod g2o_types_sba
        ${OpenCV_LIBS})