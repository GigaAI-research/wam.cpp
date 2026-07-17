include_guard(GLOBAL)

include(FetchContent)

set(WAM_LLAMA_REVISION "b9866" CACHE INTERNAL "Pinned llama.cpp revision")
set(WAM_LLAMA_ARCHIVE_SHA256
    "fb4f9e480938406dd17a27e9a3b4b99fb558685d8ad44040778062c5e999d8ae"
    CACHE INTERNAL "Pinned llama.cpp source archive SHA256")

function(wam_verify_patch path expected)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Required llama.cpp patch is missing: ${path}")
    endif()
    file(SHA256 "${path}" actual)
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR
            "llama.cpp patch hash mismatch for ${path}: expected ${expected}, got ${actual}")
    endif()
endfunction()

function(wam_configure_dependencies)
    set(fusion_patch "${PROJECT_SOURCE_DIR}/patches/llama-b9866-p7-fusions.patch")
    set(bf16_patch "${PROJECT_SOURCE_DIR}/patches/llama-b9866-native-bf16.patch")
    wam_verify_patch("${fusion_patch}"
        "4e5fb3d09bec3cee09a8dd3aeb3d31d3a7d507ba5b260a491bcee56399188f54")
    wam_verify_patch("${bf16_patch}"
        "e9f21832755d3e5425c2bd35fab3c0c0a00e667343dd01aedc4bf3d8ed146eef")

    set(GGML_CUDA ${WAM_CUDA} CACHE BOOL "Build ggml CUDA backend" FORCE)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "NCCL is not required by batch-one wam" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Enable CUDA Graph support" FORCE)
    set(GGML_CUDA_CUDNN ${WAM_CUDNN} CACHE BOOL
        "Enable the validated cuDNN BF16 convolution path" FORCE)
    set(GGML_CPU ON CACHE BOOL "Phase 5 provides the portable F32 reference backend" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "Disable ggml tests" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Disable ggml examples" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "Portable CPU metadata build" FORCE)
    set(GGML_CCACHE OFF CACHE BOOL "Do not require ccache" FORCE)

    find_program(WAM_PATCH_EXECUTABLE patch REQUIRED)

    set(patch_fusion
        ${CMAKE_COMMAND}
        -DLLAMA_SOURCE_DIR=<SOURCE_DIR>
        -DLLAMA_PATCH_FILE=${fusion_patch}
        -DLLAMA_PATCH_MARKER=GGML_CUDA_DISABLE_NORM_MODULATION_FUSION
        -DPATCH_EXECUTABLE=${WAM_PATCH_EXECUTABLE}
        -P ${PROJECT_SOURCE_DIR}/cmake/ApplyLlamaPatch.cmake)
    set(patch_bf16
        ${CMAKE_COMMAND}
        -DLLAMA_SOURCE_DIR=<SOURCE_DIR>
        -DLLAMA_PATCH_FILE=${bf16_patch}
        -DLLAMA_PATCH_MARKER=ggml_norm_modulation_bf16
        -DLLAMA_PATCH_MARKER_FILE=ggml/src/ggml.c
        -DPATCH_EXECUTABLE=${WAM_PATCH_EXECUTABLE}
        -P ${PROJECT_SOURCE_DIR}/cmake/ApplyLlamaPatch.cmake)

    if(WAM_LLAMA_SOURCE_DIR)
        get_filename_component(llama_source "${WAM_LLAMA_SOURCE_DIR}" ABSOLUTE)
        if(NOT EXISTS "${llama_source}/ggml/CMakeLists.txt")
            message(FATAL_ERROR "WAM_LLAMA_SOURCE_DIR is not a llama.cpp source tree")
        endif()
        foreach(command IN ITEMS patch_fusion patch_bf16)
            set(resolved ${${command}})
            list(TRANSFORM resolved REPLACE "<SOURCE_DIR>" "${llama_source}")
            execute_process(COMMAND ${resolved} RESULT_VARIABLE result)
            if(NOT result EQUAL 0)
                message(FATAL_ERROR "Failed to apply pinned llama.cpp patch")
            endif()
        endforeach()
        add_subdirectory("${llama_source}/ggml" "${CMAKE_BINARY_DIR}/_deps/llama-build")
        set(llama_SOURCE_DIR "${llama_source}" CACHE INTERNAL "Resolved llama source" FORCE)
    else()
        set(llama_url
            "https://github.com/ggml-org/llama.cpp/archive/refs/tags/${WAM_LLAMA_REVISION}.tar.gz")
        if(WAM_LLAMA_ARCHIVE)
            get_filename_component(llama_archive "${WAM_LLAMA_ARCHIVE}" ABSOLUTE)
            if(NOT EXISTS "${llama_archive}")
                message(FATAL_ERROR "WAM_LLAMA_ARCHIVE does not exist: ${llama_archive}")
            endif()
            set(llama_url "file://${llama_archive}")
        endif()
        FetchContent_Declare(llama
            URL "${llama_url}"
            URL_HASH "SHA256=${WAM_LLAMA_ARCHIVE_SHA256}"
            DOWNLOAD_EXTRACT_TIMESTAMP FALSE
            SOURCE_SUBDIR ggml
            PATCH_COMMAND ${patch_fusion} COMMAND ${patch_bf16})
        FetchContent_MakeAvailable(llama)
        if(NOT llama_SOURCE_DIR)
            message(FATAL_ERROR "Pinned llama.cpp source directory was not resolved")
        endif()
        # FetchContent variables are function-scoped with newer CMake releases.
        set(llama_SOURCE_DIR "${llama_SOURCE_DIR}" CACHE INTERNAL
            "Resolved llama source" FORCE)
    endif()

    if(NOT TARGET ggml)
        message(FATAL_ERROR "Pinned llama.cpp did not provide the ggml target")
    endif()
    message(STATUS "wam: llama.cpp=${WAM_LLAMA_REVISION} source=${llama_SOURCE_DIR}")
endfunction()
