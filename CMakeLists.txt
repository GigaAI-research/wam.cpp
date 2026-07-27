cmake_minimum_required(VERSION 3.22)

project(wam VERSION 0.5.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(WAM_BUILD_GWP05 "Build the GWP05 architecture entry point" ON)
option(WAM_BUILD_FASTWAM "Build the FastWAM architecture entry point" ON)
option(WAM_BUILD_TESTS "Build wam.cpp contract tests" ON)
option(WAM_BUILD_SERVING "Build the C ABI used by evaluation servers" ON)
option(WAM_CUDA "Build the pinned ggml CUDA backend" OFF)
option(WAM_CUDNN "Enable the validated GWP05 BF16 VAE path" OFF)
set(WAM_TEST_GWP05_REAL_GGUF "" CACHE FILEPATH
    "Optional real GWP05 GGUF used by the Slice 4A artifact gate")
set(WAM_TEST_GWP05_ROBOTWIN_GGUF "" CACHE FILEPATH
    "Optional formal 14D RoboTwin PolicySpec GGUF used by Gate A")
set(WAM_TEST_GWP05_REPLAY_ROOT "" CACHE PATH
    "Optional external GWP05 replay root used by the Slice 4A donor gate")
set(WAM_TEST_GWP05_STAGE_ROOT "" CACHE PATH
    "Optional external GWP05 complete-graph stage root used by the Slice 4A donor gate")
set(WAM_TEST_GWP05_CACHE_STAGE_ROOT "" CACHE PATH
    "Optional external GWP05 prefix-cache stage root used by the Slice 4A donor gate")
set(WAM_TEST_GWP05_DONOR_EXECUTABLE "" CACHE FILEPATH
    "Optional GWP05 donor executable used to identify the Slice 4A oracle")
set(WAM_TEST_GWP05_MATERIALIZED_INPUT_DIR "" CACHE PATH
    "Optional decoded PPM/F32 replay directory used by the Slice 4B engine gate")
set(WAM_TEST_GWP05_ROBOTWIN_INPUT_DIR "" CACHE PATH
    "Optional frozen 14D RoboTwin observation used by Gate A")
set(WAM_TEST_GWP05_ROBOTWIN_REFERENCE_DIR "" CACHE PATH
    "Optional independent PyTorch BF16 reference used by Gate A")
set(WAM_TEST_FASTWAM_LIBERO_GGUF "" CACHE FILEPATH
    "Optional FastWAM LIBERO PolicySpec GGUF used by Gate B")
set(WAM_TEST_FASTWAM_ROBOTWIN_GGUF "" CACHE FILEPATH
    "Optional FastWAM RoboTwin PolicySpec GGUF used by Gate B")
set(WAM_TEST_FASTWAM_LIBERO_REPLAY_DIR "" CACHE PATH
    "Optional frozen FastWAM LIBERO replay used by Gate B CUDA parity")

if(WAM_CUDNN AND NOT WAM_CUDA)
    message(FATAL_ERROR "WAM_CUDNN requires WAM_CUDA=ON")
endif()
if(WAM_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(FATAL_ERROR
            "WAM_CUDA requires explicit CMAKE_CUDA_ARCHITECTURES")
    endif()
endif()

wam_configure_gguf_dependency()

add_library(wam_core STATIC
    src/types.cpp
    src/arch.cpp
    src/model.cpp
    src/model_internal.cpp
    src/model_registry.cpp
    src/models/common/gguf_reader.cpp
    src/models/common/input_validation.cpp
    src/models/common/scheduler.cpp
    src/policy/action_ops.cpp
    src/policy/image_ops.cpp
    src/policy/language_ops.cpp
    src/policy/observation_ops.cpp
    src/policy/policy_spec.cpp
    src/policy/state_ops.cpp
)

add_library(wam::core ALIAS wam_core)
set_target_properties(wam_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(WAM_BUILD_GWP05)
    target_sources(wam_core PRIVATE
        src/models/gwp05/artifact.cpp
        src/models/gwp05/engine/action.cpp
        src/models/gwp05/engine/backbone.cpp
        src/models/gwp05/engine/cache.cpp
        src/models/gwp05/engine/engine.cpp
        src/models/gwp05/engine/language_encoder.cpp
        src/models/gwp05/engine/observation_encoder.cpp
        src/models/gwp05/engine/runtime.cpp
        src/models/gwp05/engine/weights.cpp
        src/models/gwp05/inputs.cpp
        src/models/gwp05/model.cpp
        src/models/gwp05/semantics.cpp)
    if(WAM_CUDA)
        target_link_libraries(wam_core PRIVATE CUDA::cudart)
        target_compile_definitions(wam_core PRIVATE
            GGML_USE_CUDA=1)
    endif()
    if(WAM_CUDNN)
        target_compile_definitions(wam_core PRIVATE WAM_GWP05_CUDNN=1)
    endif()
endif()

if(WAM_BUILD_FASTWAM)
    target_sources(wam_core PRIVATE
        src/models/fastwam/artifact.cpp
        src/models/fastwam/engine/action_dit.cpp
        src/models/fastwam/engine/debug.cpp
        src/models/fastwam/engine/engine.cpp
        src/models/fastwam/engine/ops.cpp
        src/models/fastwam/engine/pipeline.cpp
        src/models/fastwam/engine/scheduler.cpp
        src/models/fastwam/engine/vae.cpp
        src/models/fastwam/engine/video_dit.cpp
        src/models/fastwam/inputs.cpp
        src/models/fastwam/model.cpp
        src/models/fastwam/semantics.cpp)
endif()

target_include_directories(wam_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(wam_core PRIVATE
    WAM_BUILD_GWP05=$<BOOL:${WAM_BUILD_GWP05}>
    WAM_BUILD_FASTWAM=$<BOOL:${WAM_BUILD_FASTWAM}>
    WAM_HAS_CUDA=$<BOOL:${WAM_CUDA}>)

target_link_libraries(wam_core PRIVATE ggml ggml-cpu)
if(WAM_CUDA)
    target_link_libraries(wam_core PRIVATE ggml-cuda)
endif()

target_compile_options(wam_core PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall;-Wextra;-Wpedantic;-Werror>)

if(WAM_BUILD_SERVING)
    add_library(wam_c_api SHARED
        src/serving/c_api.cpp
        src/serving/protocol_adapter.cpp)
    add_library(wam::c_api ALIAS wam_c_api)
    target_link_libraries(wam_c_api PRIVATE wam::core)
    target_include_directories(wam_c_api
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(wam_c_api PRIVATE WAM_C_API_BUILD=1)
    target_compile_options(wam_c_api PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
    set_target_properties(wam_c_api PROPERTIES OUTPUT_NAME wam_c_api)

    find_program(WAM_PROTOC_EXECUTABLE protoc)
    if(WAM_PROTOC_EXECUTABLE)
        set(WAM_PROTO_DESCRIPTOR "${CMAKE_BINARY_DIR}/wam.desc")
        add_custom_command(
            OUTPUT "${WAM_PROTO_DESCRIPTOR}"
            COMMAND "${WAM_PROTOC_EXECUTABLE}"
                --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
                --include_imports
                --descriptor_set_out=${WAM_PROTO_DESCRIPTOR}
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/wam.proto
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/proto/wam.proto"
            VERBATIM)
        add_custom_target(wam_proto_descriptor ALL
            DEPENDS "${WAM_PROTO_DESCRIPTOR}")
    else()
        message(WARNING "protoc was not found; wam.desc will not be generated")
    endif()
endif()

if(WAM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
