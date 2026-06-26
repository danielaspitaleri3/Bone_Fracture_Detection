# CMakeLists.txt for IPA executable

# Include libraries
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/ipalib)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/ucaslib)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find sources inside IPA folder
file(GLOB src
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Fix va_start error in Visual Studio
add_definitions(-D_CRT_NO_VA_START_VALIDATION)

# Create executable
add_executable(ipaproject ${src})

# Link executable to libraries
target_link_libraries(ipaproject
    ipalib
    ucaslib
    ${OpenCV_LIBS}
)