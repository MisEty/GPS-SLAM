##########################
# SetCUDAAppTarget.cmake #
##########################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)

ADD_EXECUTABLE(${targetname} ${sources} ${headers})

