#################
# UseCUDA.cmake #
#################

enable_language(CUDA)

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA Toolkit found")
else()
  message(WARNING "CUDA Toolkit not found!")
endif()

OPTION(WITH_CUDA "Build with CUDA support?" ${CUDAToolkit_FOUND})

IF(WITH_CUDA)
  message(STATUS "Building with CUDA support")
  SET(CMAKE_CUDA_ARCHITECTURES 70;75;80)

  # Auto-detect the CUDA compute capability.
  # SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  # IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
  #   INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
  # ENDIF()

  # # Set the compute capability flags.
  # FOREACH(compute_capability ${CUDA_COMPUTE_CAPABILITY})
  #   LIST(APPEND CUDA_NVCC_FLAGS --generate-code arch=compute_${compute_capability},code=sm_${compute_capability})
  # ENDFOREACH()
  # message(STATUS "CUDA compute capability: ${CUDA_COMPUTE_CAPABILITY}")

  # Enable fast math.
  SET(CUDA_NVCC_FLAGS --use_fast_math ; ${CUDA_NVCC_FLAGS})

  message(STATUS "CUDA NVCC Flags: ${CUDA_NVCC_FLAGS}")
ELSE()
  message(STATUS "Building without CUDA support")
  ADD_DEFINITIONS(-DCOMPILE_WITHOUT_CUDA)
ENDIF()