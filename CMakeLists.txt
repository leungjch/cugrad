cmake_minimum_required(VERSION 3.18...3.22)  # Specify a range to handle policies
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
include_directories(${Python3_INCLUDE_DIRS} include)

# Add Tensor source files
set(CUGRAD_SOURCES src/tensor.cpp src/op.cpp src/nn.cpp)

# Add bindings using pybind11
pybind11_add_module(cugrad src/bindings.cpp ${CUGRAD_SOURCES})

# # Define the install target
# install(TARGETS cugrad
#         LIBRARY DESTINATION .)
install(TARGETS cugrad
    LIBRARY DESTINATION .)
