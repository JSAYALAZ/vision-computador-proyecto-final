cmake_minimum_required(VERSION 3.16)
project(plumilla_desktop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # Release: si no, los tiempos de CPU salen inflados
endif()

# OpenCV debe estar compilado con OpenCL (WITH_OPENCL=ON) y el módulo viz
# de opencv_contrib (WITH_VTK=ON, BUILD_opencv_viz=ON). Si OpenCV está en una
# ruta propia:  cmake -DOpenCV_DIR=/ruta/a/opencv/build ..
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio viz)

message(STATUS "OpenCV ${OpenCV_VERSION} en ${OpenCV_DIR}")

add_executable(plumilla_desktop main.cpp)
target_include_directories(plumilla_desktop PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(plumilla_desktop PRIVATE ${OpenCV_LIBS})
if(WIN32)
  target_link_libraries(plumilla_desktop PRIVATE psapi)
endif()
