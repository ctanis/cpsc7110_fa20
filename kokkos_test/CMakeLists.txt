cmake_minimum_required(VERSION 3.17)

IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  CMAKE_POLICY(SET CMP0074 NEW)
ENDIF()

project(kokkos-test)
find_package(Kokkos)

add_executable(kokkos-test kokkos_test.cpp)
target_link_libraries(kokkos-test Kokkos::kokkos)

SET( CMAKE_CXX_COMPILER ${KOKKOS_CXX_COMPILER} )
