cmake_minimum_required(VERSION 3.22.1)
project("birdgraph")

add_library(birdgraph SHARED
        dsp.cpp
        graph.cpp
        jni_bridge.cpp)

target_compile_options(birdgraph PRIVATE
        -O3 -ffast-math -funroll-loops -Wall)

find_library(log-lib log)
target_link_libraries(birdgraph ${log-lib})
