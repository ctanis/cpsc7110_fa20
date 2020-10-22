cmake_minimum_required(VERSION 3.13)

IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

project(kokkos_mpi_test)
find_package(Kokkos)

SET( CMAKE_CXX_COMPILER "mpic++" )
add_executable(kokkos_mpi_test kokkos_mpi_test.cpp)
target_link_libraries(kokkos_mpi_test Kokkos::kokkos)


message("set this variable before running make!")
message("export OMPI_CXX=" ${Kokkos_CXX_COMPILER})


