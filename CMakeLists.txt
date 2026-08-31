cmake_minimum_required(VERSION 3.27)

project(_rmsnorm_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
list(PREPEND CMAKE_PREFIX_PATH "${MLX_ROOT}")
find_package(MLX CONFIG REQUIRED)

add_library(smolvla_rmsnorm STATIC smolvla_mlx/native/rmsnorm.cpp)
target_include_directories(smolvla_rmsnorm PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(smolvla_rmsnorm PUBLIC mlx)
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
  target_compile_options(smolvla_rmsnorm PRIVATE -ffp-contract=off -fno-fast-math)
endif()

nanobind_add_module(
  _rmsnorm_native
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  smolvla_mlx/native/bindings.cpp)
target_link_libraries(_rmsnorm_native PRIVATE smolvla_rmsnorm mlx)
