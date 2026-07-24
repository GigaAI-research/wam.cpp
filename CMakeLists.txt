cmake_minimum_required(VERSION 3.22)

project(wam_phase1_skeleton VERSION 0.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(wam_phase1_skeleton STATIC
    src/arch.cpp
    src/model.cpp
    src/model_internal.cpp
    src/model_registry.cpp
    src/policy/policy_spec.cpp
    src/policy/image_ops.cpp
    src/policy/state_ops.cpp
    src/policy/action_ops.cpp
    src/models/common/gguf_reader.cpp
    src/models/gwp05/artifact.cpp
    src/models/gwp05/inputs.cpp
    src/models/gwp05/semantics.cpp
    src/models/gwp05/engine/engine.cpp
    src/models/gwp05/model.cpp
    src/models/fastwam/artifact.cpp
    src/models/fastwam/inputs.cpp
    src/models/fastwam/semantics.cpp
    src/models/fastwam/engine/engine.cpp
    src/models/fastwam/model.cpp
)

target_include_directories(wam_phase1_skeleton
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(wam_phase1_skeleton
    PUBLIC WAM_PHASE1_DECLARATIONS_ONLY=1)

target_compile_options(wam_phase1_skeleton PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
