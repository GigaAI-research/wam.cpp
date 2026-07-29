if(NOT DEFINED WAM_SOURCE_DIR)
    message(FATAL_ERROR "WAM_SOURCE_DIR is required")
endif()

function(wam_assert_no_match label pattern)
    set(files ${ARGN})
    foreach(path IN LISTS files)
        file(READ "${path}" contents)
        if(contents MATCHES "${pattern}")
            file(RELATIVE_PATH relative "${WAM_SOURCE_DIR}" "${path}")
            message(FATAL_ERROR "${label}: ${relative}")
        endif()
    endforeach()
endfunction()

file(GLOB public_headers "${WAM_SOURCE_DIR}/include/wam/*.h")
wam_assert_no_match(
    "public header includes an internal or GGML header"
    "#[ \t]*include[ \t]*[<\"](artifact/|backends/|models/|policy/|runtime/|ggml[^\">]*)"
    ${public_headers})

file(GLOB_RECURSE policy_sources
    "${WAM_SOURCE_DIR}/src/policy/*.h"
    "${WAM_SOURCE_DIR}/src/policy/*.cpp")
wam_assert_no_match(
    "policy code depends on a concrete model"
    "#[ \t]*include[ \t]*[<\"]models/(gwp05|fastwam)/"
    ${policy_sources})
wam_assert_no_match(
    "policy code depends on the GGUF parser instead of ArtifactView"
    "#[ \t]*include[ \t]*[<\"]artifact/gguf_reader\\.h"
    ${policy_sources})

file(GLOB_RECURSE artifact_sources
    "${WAM_SOURCE_DIR}/src/artifact/*.h"
    "${WAM_SOURCE_DIR}/src/artifact/*.cpp")
wam_assert_no_match(
    "artifact code depends on a concrete model or policy"
    "#[ \t]*include[ \t]*[<\"](models/(gwp05|fastwam)/|policy/)"
    ${artifact_sources})

file(GLOB_RECURSE gwp05_sources
    "${WAM_SOURCE_DIR}/src/models/gwp05/*.h"
    "${WAM_SOURCE_DIR}/src/models/gwp05/*.cpp")
if(EXISTS "${WAM_SOURCE_DIR}/src/models/gwp05/engine" OR
   EXISTS "${WAM_SOURCE_DIR}/src/models/gwp05/engine_internal.h")
    message(FATAL_ERROR
        "GWP05 legacy engine directory or bus header was reintroduced")
endif()
file(GLOB_RECURSE gwp05_network_sources
    "${WAM_SOURCE_DIR}/src/models/gwp05/networks/*.h"
    "${WAM_SOURCE_DIR}/src/models/gwp05/networks/*.cpp")
wam_assert_no_match(
    "GWP05 private engine namespace was reintroduced"
    "namespace[ \\t]+engine([ \\t]*\\{|[ \\t]*$)"
    ${gwp05_sources})
wam_assert_no_match(
    "GWP05 network code depends on artifact, policy, model, or serving"
    "#[ \\t]*include[ \\t]*[<\"](artifact/|policy/|model_internal|serving/|bindings/)"
    ${gwp05_network_sources})
file(GLOB_RECURSE fastwam_sources
    "${WAM_SOURCE_DIR}/src/models/fastwam/*.h"
    "${WAM_SOURCE_DIR}/src/models/fastwam/*.cpp")
if(EXISTS "${WAM_SOURCE_DIR}/src/models/fastwam/engine" OR
   EXISTS "${WAM_SOURCE_DIR}/src/models/fastwam/engine_internal.h")
    message(FATAL_ERROR
        "FastWAM legacy engine directory or bus header was reintroduced")
endif()
file(GLOB_RECURSE fastwam_network_sources
    "${WAM_SOURCE_DIR}/src/models/fastwam/networks/*.h"
    "${WAM_SOURCE_DIR}/src/models/fastwam/networks/*.cpp")
wam_assert_no_match(
    "FastWAM private engine namespace was reintroduced"
    "namespace[ \\t]+engine([ \\t]*\\{|[ \\t]*$)"
    ${fastwam_sources})
wam_assert_no_match(
    "FastWAM network code depends on artifact, policy, model, or serving"
    "#[ \\t]*include[ \\t]*[<\"](artifact/|policy/|model_internal|serving/|bindings/)"
    ${fastwam_network_sources})
wam_assert_no_match(
    "GWP05 code depends on FastWAM"
    "#[ \t]*include[ \t]*[<\"]models/fastwam/"
    ${gwp05_sources})
wam_assert_no_match(
    "FastWAM code depends on GWP05"
    "#[ \t]*include[ \t]*[<\"]models/gwp05/"
    ${fastwam_sources})
wam_assert_no_match(
    "model code reads hidden environment variables"
    "(std::)?getenv[ \\t]*\\("
    ${gwp05_sources}
    ${fastwam_sources})
wam_assert_no_match(
    "model code writes directly to stdio"
    "(std::)?f(printf|puts|write)[ \\t]*\\("
    ${gwp05_sources}
    ${fastwam_sources})

file(GLOB_RECURSE backend_sources
    "${WAM_SOURCE_DIR}/src/backends/*.h"
    "${WAM_SOURCE_DIR}/src/backends/*.cpp")
wam_assert_no_match(
    "backend code depends on a concrete model, policy, or artifact parser"
    "#[ \\t]*include[ \\t]*[<\"](models/|policy/|artifact/)"
    ${backend_sources})

set(core_model_sources
    "${WAM_SOURCE_DIR}/src/model.cpp"
    "${WAM_SOURCE_DIR}/src/model_internal.cpp"
    "${WAM_SOURCE_DIR}/src/runtime/model_registry.cpp"
    "${WAM_SOURCE_DIR}/src/models/builtin_modules.cpp"
    ${gwp05_sources}
    ${fastwam_sources})
wam_assert_no_match(
    "core/model code contains simulator-specific behavior"
    "([Rr][Oo][Bb][Oo][Tt][Ww][Ii][Nn]|[Ll][Ii][Bb][Ee][Rr][Oo])"
    ${core_model_sources})

file(GLOB_RECURSE production_sources
    "${WAM_SOURCE_DIR}/include/*.h"
    "${WAM_SOURCE_DIR}/src/*.h"
    "${WAM_SOURCE_DIR}/src/*.cpp")
wam_assert_no_match(
    "production code includes a .cpp file"
    "#[ \t]*include[ \t]*[<\"][^\">]*\\.cpp[\">]"
    ${production_sources})
