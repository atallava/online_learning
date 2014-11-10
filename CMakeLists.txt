# CMake root for Online Learning
# Version required.
cmake_minimum_required (VERSION 2.8)
# Project name. Let's call it OL.
project (OL)

set (CMAKE_BUILD_TYPE "Debug")

if( NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
  message( STATUS "No build type selected, default to Release" )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE )
endif()

message( STATUS "Build configuration: ${CMAKE_BUILD_TYPE}" )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set( LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib" CACHE PATH
    "Target directory for all libraries." )
set( EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin" CACHE PATH
    "Target directory for all executables." )

add_definitions( -Wall -std=c++0x )

# Find packages we will be using

# PCL for visualizing
find_package( PCL 1.3 REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions(${PCL_DEFINITIONS})
list( APPEND OL_LIBRARIES ${PCL_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# Eigen for linear algebra and matrices
find_package( Eigen REQUIRED )
message( STATUS "Eigen found at: ${EIGEN_INCLUDE_DIRS}" )
include_directories( ${EIGEN_INCLUDE_DIRS} )

# Threading and Boost utilities
find_package( Threads REQUIRED )
list( APPEND OL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} )

find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )
list( APPEND OL_LIBRARIES ${Boost_LIBRARIES} )
list( APPEND OL_LIBRARIES "-lboost_thread" ) # Need to force linking against threads

set( OL_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include" )
include_directories(${OL_INCLUDE_DIRECTORIES})

# Add files here to compile them
add_library(ol SHARED 	src/Dataset.cpp
						src/Visualizer.cpp

                        src/ExpGradDescent.cpp
                        src/Logistic.cpp

                        src/Test.cpp 
                  	)

set_target_properties(ol PROPERTIES OUTPUT_NAME ol)

target_link_libraries(ol ${OL_LIBRARIES})

# Add files here to make executables out of them
add_executable(run_tests src/run_tests.cpp)
target_link_libraries(run_tests ol ${OL_LIBRARIES})

# add_subdirectory( src )

# add_subdirectory( app )
