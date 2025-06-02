cmake_minimum_required(VERSION 3.21)
project(CUDA_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(MPI_LIBS mpi mpi_cxx)

find_package(CUDAToolkit REQUIRED)
add_executable(CUDA_test main.cu)

target_link_libraries(CUDA_test PRIVATE ${MPI_LIBS} CUDA::cudart)

include(FetchContent)
FetchContent_Declare(
        kamping
        GIT_REPOSITORY https://github.com/kamping-site/kamping.git
        GIT_TAG ca6a7324e49b5ca64862ddcc3a0b56475ad748fc
)

FetchContent_MakeAvailable(kamping)
target_compile_options(kamping INTERFACE
   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=gnu++17>)
target_link_libraries(CUDA_test PRIVATE kamping::kamping)