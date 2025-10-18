##########################
# SetCUDALibTarget.cmake #
##########################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)

# message(STATUS "sources ${sources}")
ADD_LIBRARY(${targetname} STATIC ${sources} ${headers} ${templates})
target_link_libraries(${targetname} PUBLIC cuda)
target_include_directories(${targetname} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)