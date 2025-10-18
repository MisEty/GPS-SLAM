###################
# UseOpenMP.cmake #
###################

OPTION(WITH_OPENMP "Enable OpenMP support?" ON)

IF(WITH_OPENMP)
  FIND_PACKAGE(OpenMP QUIET)
  IF(OPENMP_FOUND)
    message(STATUS "USE OPENMP")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ADD_DEFINITIONS(-DWITH_OPENMP)
  ELSE()
    message(WARNING "OPENMP not found!")
  ENDIF()
ENDIF()
