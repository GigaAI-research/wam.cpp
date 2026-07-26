include_guard(GLOBAL)

include(FetchContent)

set(WAM_LLAMA_REVISION "b9866" CACHE INTERNAL
    "Pinned llama.cpp revision used by the GGUF runtime")
set(WAM_LLAMA_ARCHIVE_SHA256
    "fb4f9e480938406dd17a27e9a3b4b99fb558685d8ad44040778062c5e999d8ae"
    CACHE INTERNAL "Pinned llama.cpp source archive SHA256")
set(WAM_LLAMA_SOURCE_DIR "" CACHE PATH
    "Optional existing llama.cpp source tree")
set(WAM_LLAMA_ARCHIVE "" CACHE FILEPATH
    "Optional local llama.cpp source archive")

function(wam_configure_gguf_dependency)
    set(fusion_patch
        "${PROJECT_SOURCE_DIR}/patches/llama-b9866-p7-fusions.patch")
    set(bf16_patch
        "${PROJECT_SOURCE_DIR}/patches/llama-b9866-native-bf16.patch")
    foreach(patch_file IN ITEMS "${fusion_patch}" "${bf16_patch}")
        if(NOT EXISTS "${patch_file}")
            message(FATAL_ERROR "Required llama.cpp patch is missing: ${patch_file}")
        endif()
    endforeach()
    file(SHA256 "${fusion_patch}" fusion_patch_sha256)
    file(SHA256 "${bf16_patch}" bf16_patch_sha256)
    if(NOT fusion_patch_sha256 STREQUAL
            "4e5fb3d09bec3cee09a8dd3aeb3d31d3a7d507ba5b260a491bcee56399188f54")
        message(FATAL_ERROR "Pinned llama.cpp fusion patch hash differs")
    endif()
    if(NOT bf16_patch_sha256 STREQUAL
            "e9f21832755d3e5425c2bd35fab3c0c0a00e667343dd01aedc4bf3d8ed146eef")
        message(FATAL_ERROR "Pinned llama.cpp native-BF16 patch hash differs")
    endif()

    if(WAM_CUDNN)
        find_path(wam_cudnn_v9_include cudnn_v9.h)
        find_path(wam_cudnn_unversioned_include cudnn.h)
        if(wam_cudnn_v9_include AND NOT wam_cudnn_unversioned_include)
            set(CUDNN_INCLUDE_DIR
                "${PROJECT_SOURCE_DIR}/cmake/cudnn_compat" CACHE PATH
                "Compatibility include directory for version-suffixed cuDNN 9 headers"
                FORCE)
        endif()
    endif()

    set(GGML_BUILD_TESTS OFF CACHE BOOL "Disable ggml tests" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Disable ggml examples" FORCE)
    set(GGML_CPU ON CACHE BOOL "Build the GWP05 F32 reference backend" FORCE)
    set(GGML_CUDA ${WAM_CUDA} CACHE BOOL "Build the ggml CUDA backend" FORCE)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "NCCL is not required" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Enable CUDA graph support" FORCE)
    set(GGML_CUDA_CUDNN ${WAM_CUDNN} CACHE BOOL
        "Enable the validated cuDNN BF16 convolution path" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "Build portable GGUF support" FORCE)
    set(GGML_CCACHE OFF CACHE BOOL "Do not require ccache" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build dependencies as static libraries" FORCE)
    set(GGML_STATIC OFF CACHE BOOL "Do not add a process-wide -static link flag" FORCE)
    set(GGML_ALL_WARNINGS OFF CACHE BOOL "Keep third-party warnings isolated" FORCE)

    if(WAM_LLAMA_SOURCE_DIR)
        get_filename_component(llama_source "${WAM_LLAMA_SOURCE_DIR}" ABSOLUTE)
        if(NOT EXISTS "${llama_source}/ggml/CMakeLists.txt")
            message(FATAL_ERROR
                "WAM_LLAMA_SOURCE_DIR is not a llama.cpp source tree: ${llama_source}")
        endif()
        find_program(WAM_PATCH_EXECUTABLE patch REQUIRED)
        execute_process(
            COMMAND ${CMAKE_COMMAND}
                -DLLAMA_SOURCE_DIR=${llama_source}
                -DLLAMA_PATCH_FILE=${fusion_patch}
                -DLLAMA_PATCH_MARKER=GGML_CUDA_DISABLE_NORM_MODULATION_FUSION
                -DPATCH_EXECUTABLE=${WAM_PATCH_EXECUTABLE}
                -P ${PROJECT_SOURCE_DIR}/cmake/ApplyLlamaPatch.cmake
            RESULT_VARIABLE fusion_result)
        execute_process(
            COMMAND ${CMAKE_COMMAND}
                -DLLAMA_SOURCE_DIR=${llama_source}
                -DLLAMA_PATCH_FILE=${bf16_patch}
                -DLLAMA_PATCH_MARKER=ggml_norm_modulation_bf16
                -DLLAMA_PATCH_MARKER_FILE=ggml/src/ggml.c
                -DPATCH_EXECUTABLE=${WAM_PATCH_EXECUTABLE}
                -P ${PROJECT_SOURCE_DIR}/cmake/ApplyLlamaPatch.cmake
            RESULT_VARIABLE bf16_result)
        if(NOT fusion_result EQUAL 0 OR NOT bf16_result EQUAL 0)
            message(FATAL_ERROR "Failed to apply pinned llama.cpp engine patches")
        endif()
        add_subdirectory("${llama_source}/ggml"
            "${CMAKE_BINARY_DIR}/_deps/llama-build")
        set(llama_SOURCE_DIR "${llama_source}" CACHE INTERNAL
            "Resolved llama.cpp source" FORCE)
    else()
        set(llama_url
            "https://github.com/ggml-org/llama.cpp/archive/refs/tags/${WAM_LLAMA_REVISION}.tar.gz")
        if(WAM_LLAMA_ARCHIVE)
            get_filename_component(llama_archive "${WAM_LLAMA_ARCHIVE}" ABSOLUTE)
            if(NOT EXISTS "${llama_archive}")
                message(FATAL_ERROR
                    "WAM_LLAMA_ARCHIVE does not exist: ${llama_archive}")
            endif()
            set(llama_url "file://${llama_archive}")
        endif()
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
        FetchContent_Declare(llama
            URL "${llama_url}"
            URL_HASH "SHA256=${WAM_LLAMA_ARCHIVE_SHA256}"
            DOWNLOAD_EXTRACT_TIMESTAMP FALSE
            SOURCE_SUBDIR ggml
            PATCH_COMMAND ${patch_fusion} COMMAND ${patch_bf16})
        FetchContent_MakeAvailable(llama)
        set(llama_SOURCE_DIR "${llama_SOURCE_DIR}" CACHE INTERNAL
            "Resolved llama.cpp source" FORCE)
    endif()

    if(NOT TARGET ggml)
        message(FATAL_ERROR "Pinned llama.cpp did not provide the ggml target")
    endif()
    message(STATUS
        "wam: GGUF dependency llama.cpp=${WAM_LLAMA_REVISION} source=${llama_SOURCE_DIR}")
endfunction()
