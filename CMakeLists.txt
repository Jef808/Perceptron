cmake_minimum_required( VERSION 3.20  )
project( Perceptron LANGUAGE CXX)

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED 17 )
set( CMAKE_EXPORT_COMPILE_COMMANDS on )

add_executable( perceptron main.cpp perceptron.cpp )
target_include_directories( perceptron ${CMAKE_CURRENT_SOURCE_DIR} )
