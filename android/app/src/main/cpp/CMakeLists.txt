cmake_minimum_required(VERSION 3.22.1)
project(plumilla LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# OpenCV_DIR se pasa desde app/build.gradle:
#   -DOpenCV_DIR=<OpenCV-android-sdk>/sdk/native/jni
find_package(OpenCV REQUIRED core imgproc)

add_library(plumilla SHARED native-lib.cpp)

# Header compartido con la app de escritorio (proyecto_integrador_p1/common)
target_include_directories(plumilla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../common)

target_link_libraries(plumilla ${OpenCV_LIBS} log)
