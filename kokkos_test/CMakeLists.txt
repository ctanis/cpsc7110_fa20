cmake_minimum_required(VERSION 3.12)

IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

project(kokkos-test)
find_package(Kokkos)
SET( CMAKE_CXX_COMPILER "${Kokkos_CXX_COMPILER}" )

add_executable(kokkos_test kokkos_test.cpp)
target_link_libraries(kokkos_test Kokkos::kokkos)


