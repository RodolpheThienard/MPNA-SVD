cmake_minimum_required(VERSION 3.10)

project(SVD)

add_executable(
  seq
  src/seq.c
  src/utils.c
  src/kernels.c
)

add_executable(
  par
  src/par.c
  src/utils.c
  src/kernels.c
)

target_link_libraries(
  seq
  -lm
  -lblas
)
target_link_libraries(
  par
  -lm
  -lblas
)

if (DEBUG)
  add_definitions(-DDEBUG)
endif()
if (BENCHMARK)
  add_definitions(-DBENCHMARK)
endif()

