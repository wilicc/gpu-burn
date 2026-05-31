cmake_minimum_required(VERSION 3.18)
project(gpu_burn_win LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default compute capability (can be overridden)
if(NOT DEFINED COMPUTE)
    set(COMPUTE "75" CACHE STRING "CUDA Compute Capability")
endif()

# Check if COMPUTE is a list of architectures (contains semicolons or spaces)
string(REPLACE ";" " " COMPUTE_LIST "${COMPUTE}")
separate_arguments(COMPUTE_ARGS WINDOWS_COMMAND "${COMPUTE_LIST}")
list(LENGTH COMPUTE_ARGS COMPUTE_LEN)

set(NVCC_FLAGS "-fatbin")
if(COMPUTE_LEN GREATER 1)
    foreach(ARCH IN LISTS COMPUTE_ARGS)
        list(APPEND NVCC_FLAGS "-gencode=arch=compute_${ARCH},code=sm_${ARCH}")
    endforeach()
    message(STATUS "Building multi-architecture fatbin: ${COMPUTE}")
else()
    if(COMPUTE MATCHES "^[0-9]+$")
        set(COMPUTE_ARCH "compute_${COMPUTE}")
    else()
        set(COMPUTE_ARCH "${COMPUTE}")
    endif()
    list(APPEND NVCC_FLAGS "-arch=${COMPUTE_ARCH}")
    message(STATUS "Building for CUDA Compute Capability: ${COMPUTE_ARCH}")
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Source files
set(SOURCES
    gpu_burn-drv.cpp
)

# Fatbin file output
set(FATBIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/compare.fatbin")

# Compile the shared CUDA kernel from the repository root
set(COMPARE_CU "${CMAKE_CURRENT_SOURCE_DIR}/../compare.cu")

add_custom_command(
    OUTPUT ${FATBIN_FILE}
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" ${NVCC_FLAGS} "${COMPARE_CU}" -o "${FATBIN_FILE}"
    DEPENDS ${COMPARE_CU}
    COMMENT "Compiling CUDA kernel to fatbin"
    VERBATIM
)

# Add executable
add_executable(gpu_burn ${SOURCES} ${FATBIN_FILE})

# Link CUDA libraries - Windows specific paths
if(WIN32)
    # Get CUDA library directory
    get_filename_component(CUDA_LIB_DIR "${CUDAToolkit_NVCC_EXECUTABLE}" DIRECTORY)
    get_filename_component(CUDA_ROOT "${CUDA_LIB_DIR}" DIRECTORY)
    
    # Add library directories
    if(EXISTS "${CUDA_ROOT}/lib/x64")
        target_link_directories(gpu_burn PRIVATE "${CUDA_ROOT}/lib/x64")
    endif()
    if(EXISTS "${CUDA_ROOT}/lib")
        target_link_directories(gpu_burn PRIVATE "${CUDA_ROOT}/lib")
    endif()
    
    # Link CUDA libraries directly
    target_link_libraries(gpu_burn PRIVATE
        cuda
        cublas
        cudart
    )
else()
    # Linux/Mac linking
    target_link_directories(gpu_burn PRIVATE
        ${CUDAToolkit_LIBRARY_DIR}
    )
    target_link_libraries(gpu_burn PRIVATE
        cuda
        cublas
        cudart
    )
endif()

# Copy fatbin file to output directory
add_custom_command(TARGET gpu_burn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${FATBIN_FILE}
    $<TARGET_FILE_DIR:gpu_burn>/compare.fatbin
    COMMENT "Copying fatbin file to output directory"
)

# Compiler flags
if(MSVC)
    target_compile_options(gpu_burn PRIVATE
        /O2
        /W3
        /EHsc
    )
    # Linker flags to suppress LIBCMT warning
    set_target_properties(gpu_burn PROPERTIES
        LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    )
else()
    target_compile_options(gpu_burn PRIVATE
        -O3
        -Wall
    )
endif()

# Include CUDA directories
target_include_directories(gpu_burn PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

