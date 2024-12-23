cmake_minimum_required(VERSION 3.18...3.22)  # Specify a range to handle policies

# Force CMake to use Debug build type
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

project(cugrad LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Include CMake modules
include(FetchContent)
include(GNUInstallDirs)  # Defines CMAKE_INSTALL_LIBDIR and others

# Set CMake policies to handle deprecation warnings
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

# Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6  # Use the installed version
)

# Make pybind11 available
FetchContent_MakeAvailable(pybind11)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${Python3_INCLUDE_DIRS} include thirdparty/tracy/public/tracy)

# Add Tensor source files
set(CUGRAD_SOURCES 
    src/tensor.cpp 
    src/op.cpp 
    src/nn.cpp 
    src/optimizer.cpp 
    src/op_cuda.cu
)

# Enable Position-Independent Code globally
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable Tracy options unconditionally
option(TRACY_ENABLE "Enable Tracy Profiler" ON)
option(TRACY_ON_DEMAND "Enable Tracy on-demand mode" ON)
option(TRACY_NO_BROADCAST "Disable broadcasting to multiple clients" ON)

# Enable Tracy to build as shared library
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)

# Add Tracy subdirectory (ensure the path is correct)
add_subdirectory(thirdparty/tracy)

# Add bindings using pybind11
pybind11_add_module(cugrad src/bindings.cpp ${CUGRAD_SOURCES})

# Link Tracy to the cugrad target
target_link_libraries(cugrad PRIVATE Tracy::TracyClient)

# Define TRACY_ENABLE macro for C++ code
target_compile_definitions(cugrad PRIVATE TRACY_ENABLE=1)

# Define the install target
install(TARGETS cugrad
    LIBRARY DESTINATION .
)
