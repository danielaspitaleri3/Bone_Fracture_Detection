# Top-Level CMakeLists.txt for Bone Fracture Detection project

cmake_minimum_required(VERSION 3.16)

project(Bone_Fracture_Detection)

# Standard C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cartelle di output
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs CACHE INTERNAL "Library output directory")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE INTERNAL "Executable output directory")

# Script CMake del progetto
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

# ============================================================
# OpenCV
# ============================================================

find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found. Set OpenCV_DIR in CMake GUI.")
endif()

# Rende gli include OpenCV visibili anche a 3rdparty, ucaslib, ipalib e IPA
include_directories(${OpenCV_INCLUDE_DIRS})

if(OpenCV_LIB_DIR)
    link_directories(${OpenCV_LIB_DIR})
endif()

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# ============================================================
# Warning Visual Studio
# ============================================================

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    warnings_disable(CMAKE_CXX_FLAGS /wd4290)
    warnings_disable(CMAKE_CXX_FLAGS /wd4996)
    warnings_disable(CMAKE_CXX_FLAGS /wd4530)
    warnings_disable(CMAKE_CXX_FLAGS /wd4503)

    add_definitions(-D_CRT_NO_VA_START_VALIDATION)
endif()

# Warning Clang
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-varargs")
endif()

# Cartella immagini di esempio, se usata nel codice
add_definitions(-DEXAMPLE_IMAGES_PATH="${PROJECT_SOURCE_DIR}/example_images")

# ============================================================
# Moduli del progetto
# ============================================================

# Prima compila le librerie esterne: ucaslib e ipalib
add_subdirectory(3rdparty)

# Poi compila il progetto principale IPA
add_subdirectory(IPA)