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

file(GLOB_RECURSE gwp05_sources
    "${WAM_SOURCE_DIR}/src/models/gwp05/*.h"
    "${WAM_SOURCE_DIR}/src/models/gwp05/*.cpp")
file(GLOB_RECURSE fastwam_sources
    "${WAM_SOURCE_DIR}/src/models/fastwam/*.h"
    "${WAM_SOURCE_DIR}/src/models/fastwam/*.cpp")
wam_assert_no_match(
    "GWP05 code depends on FastWAM"
    "#[ \t]*include[ \t]*[<\"]models/fastwam/"
    ${gwp05_sources})
wam_assert_no_match(
    "FastWAM code depends on GWP05"
    "#[ \t]*include[ \t]*[<\"]models/gwp05/"
    ${fastwam_sources})

set(core_model_sources
    "${WAM_SOURCE_DIR}/src/model.cpp"
    "${WAM_SOURCE_DIR}/src/model_internal.cpp"
    "${WAM_SOURCE_DIR}/src/model_registry.cpp"
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
