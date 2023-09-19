set(PROJECT_NAME Math_extra)                    
set(LIB_NAME math_extra)                        
set(EXTERNAL_INCLUDES )
set(STATIC_LINKS ) 
set(COMPILE_OPTIONS_ADD )  
set(LINKS )

cmake_minimum_required(VERSION 3.15)
project(${PROJECT_NAME} ) # Needs to be ontop

# Prepending the absolute path
list(TRANSFORM EXTERNAL_INCLUDES PREPEND "${PROJECT_SOURCE_DIR}/../" REGEX "^(.*)$" )
list(TRANSFORM STATIC_LINKS PREPEND "${PROJECT_SOURCE_DIR}/../" REGEX "^(.*)$" )
list(APPEND LINKS ${STATIC_LINKS})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/obj)

file(GLOB SRC_FILES "src/*.cpp")
set(SRC_STATIC ${SRC_FILES})
list(FILTER SRC_STATIC EXCLUDE REGEX ".*_py\\.cpp$")
list(FILTER SRC_STATIC EXCLUDE REGEX ".*python_submodule\\.cpp$")

add_compile_options(-Wall -O3 -march=native )
add_compile_options(${COMPILE_OPTIONS_ADD})

add_library(${LIB_NAME} STATIC ${SRC_STATIC}) #static library (i.e. to import in other c++ projects)
include(config.cmake) # sets IS_MINGW,PY_INCL,SHARED_LIB_PREFIX,SHARED_LIB_SUFFIX and updates LINKS

#compile includes
include_directories(${EXTERNAL_INCLUDES})

#linking 
set_target_properties(${LIB_NAME} PROPERTIES PREFIX "${SHARED_LIB_PREFIX}")
target_link_libraries(${LIB_NAME} ${LINKS})

