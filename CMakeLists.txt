cmake_minimum_required(VERSION 3.10)

project(SVD)

add_executable(
  Main.exe
  src/main.c
  src/svd.c
)

target_link_libraries(
  Main.exe 
  -lm
  -lblas
  -llapacke
)

if (DEBUG)
  add_definitions(-DDEBUG)
endif()
if (BENCHMARK)
  add_definitions(-DBENCHMARK)
endif()

