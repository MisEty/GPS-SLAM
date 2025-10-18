cmake_minimum_required(VERSION 3.21)
project(opensplat)

set(GPU_RUNTIME "CUDA" )
set(OPENSPLAT_MAX_CUDA_COMPATIBILITY OFF CACHE BOOL "Build for maximum CUDA device compatibility")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/ThirdLibs/libtorch ${PROJECT_SOURCE_DIR}/ThirdLibs/install)

# Read version
file(READ "VERSION" APP_VERSION)

# Read git commit
set(GIT_REV "")
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" 
                OUTPUT_VARIABLE GIT_REV 
                ERROR_QUIET)
string(REGEX REPLACE "\n$" "" GIT_REV "${GIT_REV}")
if (NOT "${GIT_REV}" STREQUAL "")
    set(DAPP_VERSION "${APP_VERSION} (git commit ${GIT_REV})")
    set(DAPP_REVISION "${GIT_REV}")
else()
    set(DAPP_VERSION "${APP_VERSION}")
    set(DAPP_REVISION "dev")
endif()

message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

add_compile_options("-DAPP_VERSION=\"${DAPP_VERSION}\"")
add_compile_options("-DAPP_REVISION=\"${DAPP_REVISION}\"")

# Don't complain about the override from NANOFLANN_BUILD_EXAMPLES
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# Use time-of-extraction for FetchContent'ed files modification time
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
# Suppress warning #20012-D (nvcc and glm)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -diag-suppress=20012)

include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_Declare(nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip
)
FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip
)
FetchContent_MakeAvailable(nlohmann_json nanoflann cxxopts)
FetchContent_Declare(glm
    URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
)
FetchContent_MakeAvailable(glm)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

find_package(CUDAToolkit)
if (NOT CUDAToolkit_FOUND)
    message(WARNING "CUDA toolkit not found, building with CPU support only")
    set(GPU_RUNTIME "CPU")
else()
    if (OPENSPLAT_MAX_CUDA_COMPATIBILITY)
        execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-arch 
                OUTPUT_VARIABLE LIST_GPU_ARCH 
                ERROR_QUIET)
    endif()

    if(NOT LIST_GPU_ARCH AND OPENSPLAT_MAX_CUDA_COMPATIBILITY)
        message(WARNING "Cannot compile for max CUDA compatibility, nvcc does not support --list-gpu-arch")
        SET(OPENSPLAT_MAX_CUDA_COMPATIBILITY OFF)
    endif()
    if(NOT OPENSPLAT_MAX_CUDA_COMPATIBILITY)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            SET(CMAKE_CUDA_ARCHITECTURES 70;75;80)
        endif()
    else()
        # Build for maximum compatibility
        # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        set(CMAKE_CUDA_ARCHITECTURES "")

        # Extract list of arch and gencodes
        string(REPLACE "\r" "" LIST_GPU_ARCH ${LIST_GPU_ARCH})
        string(REPLACE "\n" ";" LIST_GPU_ARCH ${LIST_GPU_ARCH})

        execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-code 
            OUTPUT_VARIABLE LIST_GPU_CODE 
            ERROR_QUIET)
        string(REPLACE "\r" "" LIST_GPU_CODE ${LIST_GPU_CODE})
        string(REPLACE "\n" ";" LIST_GPU_CODE ${LIST_GPU_CODE})

        list(GET LIST_GPU_CODE 0 TARGET_GPU_CODE)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${TARGET_GPU_CODE}")
        
        set(IDX 0)
        foreach(GPU_ARCH ${LIST_GPU_ARCH})
            string(REGEX MATCH "compute_([0-9]+)" GPU_ARCH_VERSION "${GPU_ARCH}")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}")
            list(GET LIST_GPU_CODE ${IDX} GPU_CODE)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=${GPU_ARCH},code=${GPU_CODE}")
            math(EXPR IDX "${IDX}+1")
        endforeach()
        message("Set CUDA flags: " ${CMAKE_CUDA_FLAGS})
    endif()
    # Set torch cuda architecture list
    set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
    list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "([0-9])([0-9])" "\\1.\\2")
    string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
    message(STATUS "** Updated TORCH_CUDA_ARCH_LIST to ${TORCH_CUDA_ARCH_LIST} **")
endif()


set(CMAKE_CXX_STANDARD 17)
enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(${GPU_RUNTIME}_STANDARD 17)

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

##################### InfiniTAM ############################
INCLUDE(${PROJECT_SOURCE_DIR}/InfiniTAM/cmake/UseGLUT.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/InfiniTAM/cmake/UseOpenGL.cmake)
add_subdirectory(InfiniTAM)


##################### LIBTORCH and OpenCV ############################
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)

##################### My LIBS ############################
set(THIRDLIBS_DIR "${CMAKE_SOURCE_DIR}/ThirdLibs/install/lib")
link_directories(${THIRDLIBS_DIR})

# yaml-cpp
set(YAML_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ThirdLibs/install/include/yaml-cpp")

# tensorboard_logger
find_package(Protobuf REQUIRED)
set(TENSORBOARD_LOGGER_DIR "${CMAKE_SOURCE_DIR}/ThirdLibs/install")
find_package(tensorboard_logger REQUIRED REQUIRED HINTS ${TENSORBOARD_LOGGER_DIR})
set(Protobuf_LIBRARIES "/usr/lib/x86_64-linux-gnu/libprotobuf.so")
message(STATUS "Protobuf_VERSION: ${Protobuf_VERSION}")
message(STATUS "Protobuf_INCLUDE_DIRS: ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Protobuf_LIBRARIES: ${Protobuf_LIBRARIES}")
message(STATUS "Protobuf_PROTOC_EXECUTABLE: ${Protobuf_PROTOC_EXECUTABLE}")

# eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# indicators
find_package(indicators REQUIRED)

# tinyply
find_package(tinyply REQUIRED)
message(STATUS "tinyply_FOUND: ${tinyply_FOUND}")
message(STATUS "tinyply_INCLUDE_DIRS: ${tinyply_INCLUDE_DIR}")
message(STATUS "tinyply_LIBRARIES: ${tinyply_LIBRARY_DIR}")

find_package(Pangolin 0.8 REQUIRED)

find_library(NVML_LIBRARY nvidia-ml PATHS /usr/lib/x86_64-linux-gnu)

##############################################################
#####################   gsplat rasterizer
set(RASTERIZER_LIBS rasterizer)
file(GLOB RASTERIZER_SRC_FILES ${PROJECT_SOURCE_DIR}/gsplat/rasterizer/*.cu ${PROJECT_SOURCE_DIR}/gsplat/rasterizer/*.cuh)

add_library(rasterizer ${RASTERIZER_SRC_FILES})
target_link_libraries(rasterizer PUBLIC cuda)
target_include_directories(rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(rasterizer PUBLIC glm::glm-header-only)
set_target_properties(rasterizer PROPERTIES LINKER_LANGUAGE CXX)


##############################################################
#####################   gsplat c++ binding
file(GLOB GSPLAT_SRC_FILES ${PROJECT_SOURCE_DIR}/gsplat/*.cpp)
add_library(gsplat ${GSPLAT_SRC_FILES})
install(TARGETS gsplat DESTINATION bin)
set_property(TARGET gsplat PROPERTY CXX_STANDARD 17)
target_include_directories(gsplat PRIVATE
    ${PROJECT_SOURCE_DIR}/gsplat/rasterizer
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(gsplat PUBLIC ${STDPPFS_LIBRARY} ${GPU_LIBRARIES} ${RASTERIZER_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

target_link_libraries(gsplat PUBLIC
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann
)
if (NOT WIN32)
    target_link_libraries(gsplat PUBLIC pthread)
endif()
target_compile_definitions(gsplat PRIVATE USE_CUDA)


file(GLOB OURS_SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB SLAM_SRC_FILES ${PROJECT_SOURCE_DIR}/slam/*.cpp ${PROJECT_SOURCE_DIR}/slam/TsdfFusion/*.cpp)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")

##############################################################
#####################   ours implementation of slam train
add_executable(slam_trainer slam_trainer.cpp ${OURS_SRC_FILES} ${SLAM_SRC_FILES})
target_link_libraries(slam_trainer PUBLIC
        gsplat
        Eigen3::Eigen
        indicators::indicators
        tinyply
        tensorboard_logger
        libyaml-cpp.a
        ${Protobuf_LIBRARIES}
        InputSource 
        ITMLib 
        MiniSlamGraphLib 
        ORUtils 
        FernRelocLib 
        cuda
        ${NVML_LIBRARY}
        ${OPENGL_LIBRARY}
        glut
)
target_include_directories(slam_trainer PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/slam
    ${PROJECT_SOURCE_DIR}/gsplat
    ${PROJECT_SOURCE_DIR}/InfiniTAM
)

#####################   ours implementation of remote viewer
add_executable(remote_viewer remote_viewer.cpp ${OURS_SRC_FILES} ${SLAM_SRC_FILES})
target_link_libraries(remote_viewer PUBLIC
        gsplat
        Eigen3::Eigen
        indicators::indicators
        tinyply
        tensorboard_logger
        libyaml-cpp.a
        ${Protobuf_LIBRARIES}
        InputSource 
        ITMLib 
        MiniSlamGraphLib 
        ORUtils 
        FernRelocLib 
        cuda
        ${NVML_LIBRARY}
        ${OPENGL_LIBRARY}
        glut
)
target_include_directories(remote_viewer PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/slam
    ${PROJECT_SOURCE_DIR}/gsplat
    ${PROJECT_SOURCE_DIR}/InfiniTAM
)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    file(GLOB OPENCV_DLL "${OPENCV_DIR}/x64/vc16/bin/opencv_world490.dll")
    set(DLLS_TO_COPY ${TORCH_DLLS} ${OPENCV_DLL})
    add_custom_command(TARGET opensplat
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLLS_TO_COPY}
        $<TARGET_FILE_DIR:opensplat>)
endif (MSVC)

add_compile_definitions(GLOG_USE_GLOG_EXPORT)