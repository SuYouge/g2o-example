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

include_directories(${CSPARSE_INCLUDE_DIR}
    ${QGLVIEWER_INCLUDE_DIR}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Gui_INCLUDE_DIRS}
    ${Qt5Xml_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5OpenGL_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
    )

QT5_WRAP_UI(UI_HEADERS base_main_window.ui)
QT5_WRAP_CPP(UI_SOURCES main_window.h)

add_executable(slam2d_g2o
  main_window.cpp
  slam2d_viewer.cpp
  slam2d_viewer.h
  slam2d_g2o.cpp
  draw_helpers.cpp
  ${UI_HEADERS}
  ${UI_SOURCES}
)

set_target_properties(slam2d_g2o PROPERTIES OUTPUT_NAME slam2d_g2o${EXE_POSTFIX})

if(Qt5_POSITION_INDEPENDENT_CODE)
    set_property(TARGET slam2d_g2o PROPERTY COMPILE_FLAGS -fPIC)
    message(STATUS "Generating position indpendent code for slam2d because Qt5 was built with -reduce-relocations")
    # Note: using
    #       set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    #       does not seem to work: This generates some libraries with -fPIE which is not enough for Qt...
endif()




target_link_libraries(slam2d_g2o g2o_core g2o_solver_csparse g2o_types_slam2d g2o_stuff OpenGL GLU g2o_csparse_extension cxsparse
    ${QGLVIEWER_LIBRARY}
    ${Qt5Core_LIBRARIES}
    ${Qt5Gui_LIBRARIES}
    ${Qt5Xml_LIBRARIES}
    ${Qt5Widgets_LIBRARIES}
    ${Qt5OpenGL_LIBRARIES}
    ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
)

