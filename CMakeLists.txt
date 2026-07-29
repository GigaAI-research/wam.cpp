cmake_minimum_required(VERSION 3.22)

project(wam VERSION 0.6.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CMakePackageConfigHelpers)
include(Dependencies)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(WAM_BUILD_GWP05 "Build the GWP05 architecture entry point" ON)
option(WAM_BUILD_FASTWAM "Build the FastWAM architecture entry point" ON)
option(WAM_BUILD_TESTS "Build wam.cpp contract tests" ON)
option(WAM_BUILD_SERVING "Build the C ABI used by evaluation servers" ON)
option(WAM_BUILD_APPS "Build artifact inspection and validation tools" ON)
option(WAM_CUDA "Build the pinned ggml CUDA backend" OFF)
option(WAM_CUDNN "Enable the validated GWP05 BF16 VAE path" OFF)
set(WAM_TEST_GWP05_REAL_GGUF "" CACHE FILEPATH
    "Optional real GWP05 GGUF used by the external artifact gate")
set(WAM_TEST_GWP05_ROBOTWIN_GGUF "" CACHE FILEPATH
    "Optional formal 14D RoboTwin PolicySpec GGUF used by Gate A")
set(WAM_TEST_GWP05_REPLAY_ROOT "" CACHE PATH
    "Optional external GWP05 replay root used by the donor parity gate")
set(WAM_TEST_GWP05_STAGE_ROOT "" CACHE PATH
    "Optional external GWP05 complete-graph stage root used by the donor parity gate")
set(WAM_TEST_GWP05_CACHE_STAGE_ROOT "" CACHE PATH
    "Optional external GWP05 prefix-cache stage root used by the donor parity gate")
set(WAM_TEST_GWP05_DONOR_EXECUTABLE "" CACHE FILEPATH
    "Optional GWP05 donor executable used to identify the frozen oracle")
set(WAM_TEST_GWP05_MATERIALIZED_INPUT_DIR "" CACHE PATH
    "Optional decoded PPM/F32 replay directory used by the engine parity gate")
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

# Internal OBJECT targets expose ownership and dependency boundaries without
# changing the single-library deployment model during the 0.6 migration.
function(wam_configure_internal_target target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE
        WAM_BUILD_GWP05=$<BOOL:${WAM_BUILD_GWP05}>
        WAM_BUILD_FASTWAM=$<BOOL:${WAM_BUILD_FASTWAM}>
        WAM_HAS_CUDA=$<BOOL:${WAM_CUDA}>)
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall;-Wextra;-Wpedantic;-Werror>)
endfunction()

add_library(wam_backend_ggml INTERFACE)
add_library(wam::backend_ggml ALIAS wam_backend_ggml)
set_target_properties(wam_backend_ggml PROPERTIES EXPORT_NAME backend_ggml)
target_link_libraries(wam_backend_ggml INTERFACE
    $<BUILD_INTERFACE:ggml>
    $<BUILD_INTERFACE:ggml-cpu>
    $<INSTALL_INTERFACE:ggml::ggml>)
if(WAM_CUDA)
    target_link_libraries(wam_backend_ggml INTERFACE
        $<BUILD_INTERFACE:ggml-cuda>)
endif()

add_library(wam_backend_ggml_impl OBJECT
    src/backends/ggml/backend_context.cpp
    src/backends/ggml/debug_dump.cpp
    src/backends/ggml/graph_context.cpp
    src/backends/ggml/graph_ops.cpp
    src/backends/ggml/tensor_io.cpp
    src/backends/ggml/weight_store.cpp)
wam_configure_internal_target(wam_backend_ggml_impl)
target_link_libraries(wam_backend_ggml_impl PRIVATE wam_backend_ggml)

add_library(wam_artifact OBJECT
    src/artifact/artifact_view.cpp
    src/artifact/gguf_reader.cpp
    src/artifact/manifest.cpp
    src/artifact/tensor_spec.cpp)
add_library(wam::artifact ALIAS wam_artifact)
wam_configure_internal_target(wam_artifact)
target_link_libraries(wam_artifact PRIVATE wam::backend_ggml)

add_library(wam_policy OBJECT
    src/policy/tensor_input.cpp
    src/policy/action_decoder.cpp
    src/policy/action_noise.cpp
    src/policy/image_ops.cpp
    src/policy/language_ops.cpp
    src/policy/observation_processor.cpp
    src/policy/policy_spec.cpp
    src/policy/state_ops.cpp)
add_library(wam::policy ALIAS wam_policy)
wam_configure_internal_target(wam_policy)
target_link_libraries(wam_policy PRIVATE
    wam_artifact
    wam::backend_ggml)

add_library(wam_runtime_base OBJECT
    src/error.cpp
    src/observation.cpp
    src/prediction.cpp
    src/model_internal.cpp
    src/runtime/logger.cpp
    src/runtime/model_registry.cpp
    src/runtime/telemetry.cpp)
add_library(wam::runtime_base ALIAS wam_runtime_base)
wam_configure_internal_target(wam_runtime_base)
target_link_libraries(wam_runtime_base PRIVATE wam_policy)

set(wam_model_objects "")
if(WAM_BUILD_GWP05)
    add_library(wam_model_gwp05 OBJECT
        src/models/common/scheduler.cpp
        src/models/gwp05/contract.cpp
        src/models/gwp05/inputs.cpp
        src/models/gwp05/module.cpp
        src/models/gwp05/model.cpp
        src/models/gwp05/pipeline.cpp
        src/models/gwp05/state.cpp
        src/models/gwp05/cache.cpp
        src/models/gwp05/cache_action.cpp
        src/models/gwp05/resources.cpp
        src/models/gwp05/runtime.cpp
        src/models/gwp05/semantics.cpp
        src/models/gwp05/networks/umt5.cpp
        src/models/gwp05/networks/vision_vae.cpp
        src/models/gwp05/networks/mot.cpp)
    add_library(wam::model_gwp05 ALIAS wam_model_gwp05)
    wam_configure_internal_target(wam_model_gwp05)
    target_link_libraries(wam_model_gwp05 PRIVATE
        wam_runtime_base
        wam_artifact
        wam_policy
        wam::backend_ggml)
    if(WAM_CUDA)
        target_link_libraries(wam_model_gwp05 PRIVATE CUDA::cudart)
        target_compile_definitions(wam_model_gwp05 PRIVATE GGML_USE_CUDA=1)
    endif()
    if(WAM_CUDNN)
        target_compile_definitions(wam_model_gwp05 PRIVATE WAM_GWP05_CUDNN=1)
    endif()
    list(APPEND wam_model_objects $<TARGET_OBJECTS:wam_model_gwp05>)
endif()

if(WAM_BUILD_FASTWAM)
    add_library(wam_model_fastwam OBJECT
        src/models/fastwam/contract.cpp
        src/models/fastwam/inputs.cpp
        src/models/fastwam/module.cpp
        src/models/fastwam/model.cpp
        src/models/fastwam/pipeline.cpp
        src/models/fastwam/scheduler.cpp
        src/models/fastwam/semantics.cpp
        src/models/fastwam/state.cpp
        src/models/fastwam/networks/action_dit.cpp
        src/models/fastwam/networks/ops.cpp
        src/models/fastwam/networks/proprio_projector.cpp
        src/models/fastwam/networks/video_dit.cpp
        src/models/fastwam/networks/vision_vae.cpp)
    add_library(wam::model_fastwam ALIAS wam_model_fastwam)
    wam_configure_internal_target(wam_model_fastwam)
    target_link_libraries(wam_model_fastwam PRIVATE
        wam_runtime_base
        wam_artifact
        wam_policy
        wam::backend_ggml)
    list(APPEND wam_model_objects $<TARGET_OBJECTS:wam_model_fastwam>)
endif()

add_library(wam_core_archive STATIC
    src/model.cpp
    src/pipeline.cpp
    src/models/builtin_modules.cpp
    $<TARGET_OBJECTS:wam_backend_ggml_impl>
    $<TARGET_OBJECTS:wam_artifact>
    $<TARGET_OBJECTS:wam_policy>
    $<TARGET_OBJECTS:wam_runtime_base>
    ${wam_model_objects})
wam_configure_internal_target(wam_core_archive)
if(WAM_CUDA AND WAM_BUILD_GWP05)
    target_link_libraries(wam_core_archive PRIVATE CUDA::cudart)
endif()
set_target_properties(wam_core_archive PROPERTIES
    EXPORT_NAME core_archive
    OUTPUT_NAME wam
    VERSION ${PROJECT_VERSION})

add_library(wam_core INTERFACE)
target_include_directories(wam_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(wam_core INTERFACE wam_core_archive)
set_target_properties(wam_core PROPERTIES EXPORT_NAME core)

add_library(wam_core_build INTERFACE)
add_library(wam::core ALIAS wam_core_build)
target_link_libraries(wam_core_build INTERFACE
    wam_core
    wam_backend_ggml)

if(WAM_BUILD_SERVING)
    add_library(wam_c_api SHARED
        src/bindings/c_api.cpp
        src/bindings/metadata_codec.cpp)
    add_library(wam::c_api ALIAS wam_c_api)
    target_link_libraries(wam_c_api PRIVATE wam::core)
    target_include_directories(wam_c_api
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(wam_c_api PRIVATE WAM_C_API_BUILD=1)
    target_compile_options(wam_c_api PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
    set_target_properties(wam_c_api PROPERTIES
        EXPORT_NAME c_api
        OUTPUT_NAME wam_c_api
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})

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

if(WAM_BUILD_APPS)
    foreach(app IN ITEMS inspect validate)
        add_executable(wam-${app} apps/${app}.cpp)
        target_link_libraries(wam-${app} PRIVATE wam::core)
        target_include_directories(wam-${app} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/apps
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_compile_features(wam-${app} PRIVATE cxx_std_17)
        target_compile_options(wam-${app} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
        install(TARGETS wam-${app}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endforeach()
endif()

set(wam_install_targets wam_core wam_core_archive)
if(WAM_BUILD_SERVING)
    list(APPEND wam_install_targets wam_c_api)
endif()
install(TARGETS ${wam_install_targets}
    EXPORT wamTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/wam
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")
install(EXPORT wamTargets
    FILE wamTargets.cmake
    NAMESPACE wam::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wam)

configure_package_config_file(
    cmake/wamConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/wamConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wam)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/wamConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/wamConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/wamConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wam)

if(WAM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
