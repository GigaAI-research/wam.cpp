function(wam_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE wam::core)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

foreach(target IN ITEMS
        wam_artifact
        wam_policy
        wam_backend_ggml
        wam_backend_ggml_impl
        wam_runtime_base
        wam_core)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "required Phase 1 target is missing: ${target}")
    endif()
endforeach()
if(WAM_BUILD_GWP05 AND NOT TARGET wam_model_gwp05)
    message(FATAL_ERROR "enabled GWP05 target is missing")
endif()
if(WAM_BUILD_FASTWAM AND NOT TARGET wam_model_fastwam)
    message(FATAL_ERROR "enabled FastWAM target is missing")
endif()

add_library(wam_test_fixtures STATIC fixtures/metadata_fixture.cpp)
target_link_libraries(wam_test_fixtures PRIVATE ggml)
target_include_directories(wam_test_fixtures PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(wam_test_fixtures PRIVATE cxx_std_17)
target_compile_options(wam_test_fixtures PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)

wam_add_test(wam_public_api_contract contracts/public_api_compile.cpp)
wam_add_test(wam_header_wam_contract contracts/header_wam.cpp)
wam_add_test(wam_header_error_contract contracts/header_error.cpp)
wam_add_test(wam_header_observation_contract contracts/header_observation.cpp)
wam_add_test(wam_header_prediction_contract contracts/header_prediction.cpp)
wam_add_test(wam_header_policy_spec_contract contracts/header_policy_spec.cpp)
wam_add_test(wam_header_runtime_config_contract contracts/header_runtime_config.cpp)
wam_add_test(wam_header_model_contract contracts/header_model.cpp)
wam_add_test(wam_header_session_contract contracts/header_session.cpp)
wam_add_test(wam_header_pipeline_contract contracts/header_pipeline.cpp)
wam_add_test(wam_header_version_contract contracts/header_version.cpp)
wam_add_test(wam_header_c_api_contract contracts/header_c_api.cpp)
wam_add_test(wam_types_unit unit/types.cpp)
wam_add_test(wam_registry_unit unit/model_registry.cpp)
add_executable(wam_runtime_backend_unit unit/runtime_backend.cpp)
target_link_libraries(wam_runtime_backend_unit PRIVATE wam::core)
target_include_directories(wam_runtime_backend_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src)
target_compile_features(wam_runtime_backend_unit PRIVATE cxx_std_17)
target_compile_options(wam_runtime_backend_unit PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
add_test(NAME wam_runtime_backend_unit COMMAND wam_runtime_backend_unit)
add_executable(wam_model_lifecycle
    integration/model_lifecycle.cpp
    fixtures/fake_public_model.cpp)
target_link_libraries(wam_model_lifecycle PRIVATE wam::core)
target_include_directories(wam_model_lifecycle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src)
target_compile_features(wam_model_lifecycle PRIVATE cxx_std_17)
target_compile_options(wam_model_lifecycle PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
add_test(NAME wam_model_lifecycle COMMAND wam_model_lifecycle)
wam_add_test(wam_gguf_reader_unit unit/gguf_reader.cpp)
wam_add_test(wam_artifact_layer_unit unit/artifact_layer.cpp)
wam_add_test(wam_input_validation_unit unit/input_validation.cpp)
wam_add_test(wam_policy_spec_unit unit/policy_spec.cpp)
wam_add_test(wam_policy_spec_failures unit/policy_spec_failures.cpp)
wam_add_test(wam_policy_ops_unit unit/policy_ops.cpp)
if(WAM_BUILD_FASTWAM)
    wam_add_test(wam_model_load_metadata integration/model_load_metadata.cpp)
endif()

if(WAM_BUILD_APPS)
    add_executable(wam_cli_fixture_writer tools/write_cli_fixture.cpp)
    target_link_libraries(wam_cli_fixture_writer PRIVATE wam_test_fixtures)
    target_include_directories(wam_cli_fixture_writer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(wam_cli_fixture_writer PRIVATE cxx_std_17)
    target_compile_options(wam_cli_fixture_writer PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
    add_test(NAME wam_artifact_cli
        COMMAND ${CMAKE_COMMAND}
            -DWAM_FIXTURE_WRITER=$<TARGET_FILE:wam_cli_fixture_writer>
            -DWAM_INSPECT=$<TARGET_FILE:wam-inspect>
            -DWAM_VALIDATE=$<TARGET_FILE:wam-validate>
            -DWAM_EXPECT_VALIDATE=${WAM_BUILD_GWP05}
            -DWAM_CLI_TEST_ROOT=${PROJECT_BINARY_DIR}/tests/cli-artifact
            -P ${CMAKE_CURRENT_SOURCE_DIR}/integration/artifact_cli.cmake)
    set_tests_properties(wam_artifact_cli PROPERTIES LABELS "artifact;cli")
endif()

add_test(NAME wam_source_boundaries
    COMMAND ${CMAKE_COMMAND}
        -DWAM_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/contracts/source_boundaries.cmake)
set_tests_properties(wam_source_boundaries PROPERTIES LABELS "contract;boundary")

add_test(NAME wam_install_consumer
    COMMAND ${CMAKE_COMMAND}
        -DWAM_BINARY_DIR=${PROJECT_BINARY_DIR}
        -DWAM_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/consumer
        -DWAM_TEST_CONFIG=$<CONFIG>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/contracts/install_consumer.cmake)
set_tests_properties(wam_install_consumer PROPERTIES
    LABELS "contract;install"
    RUN_SERIAL TRUE
    TIMEOUT 180)

if(WAM_BUILD_GWP05)
    wam_add_test(wam_gwp05_artifact integration/gwp05_artifact.cpp)
    wam_add_test(wam_gwp05_pipeline unit/gwp05_pipeline.cpp)
    wam_add_test(wam_gwp05_semantics unit/gwp05_semantics.cpp)
    wam_add_test(wam_gwp05_inputs unit/gwp05_inputs.cpp)

    add_executable(wam_gwp05_real_artifact
        integration/gwp05_real_artifact.cpp)
    target_link_libraries(wam_gwp05_real_artifact PRIVATE wam::core)
    target_include_directories(wam_gwp05_real_artifact PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(wam_gwp05_real_artifact PRIVATE cxx_std_17)
    target_compile_options(wam_gwp05_real_artifact PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)

    add_executable(wam_gwp05_robotwin_artifact
        integration/gwp05_robotwin_artifact.cpp)
    target_link_libraries(wam_gwp05_robotwin_artifact PRIVATE wam::core)
    target_include_directories(wam_gwp05_robotwin_artifact PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(wam_gwp05_robotwin_artifact PRIVATE cxx_std_17)
    target_compile_options(wam_gwp05_robotwin_artifact PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)

    add_executable(wam_gwp05_robotwin_parity
        integration/gwp05_robotwin_parity.cpp)
    target_link_libraries(wam_gwp05_robotwin_parity PRIVATE wam::core)
    target_include_directories(wam_gwp05_robotwin_parity PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(wam_gwp05_robotwin_parity PRIVATE cxx_std_17)
    target_compile_options(wam_gwp05_robotwin_parity PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)

    add_executable(wam_gwp05_pipeline_reference
        integration/gwp05_pipeline_reference.cpp)
    target_link_libraries(wam_gwp05_pipeline_reference PRIVATE wam::core)
    target_include_directories(wam_gwp05_pipeline_reference PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(wam_gwp05_pipeline_reference PRIVATE cxx_std_17)
    target_compile_options(wam_gwp05_pipeline_reference PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)

    if(WAM_TEST_GWP05_REAL_GGUF)
        if(NOT IS_ABSOLUTE "${WAM_TEST_GWP05_REAL_GGUF}")
            message(FATAL_ERROR
                "WAM_TEST_GWP05_REAL_GGUF must be an absolute path")
        endif()
        if(NOT EXISTS "${WAM_TEST_GWP05_REAL_GGUF}")
            message(FATAL_ERROR
                "WAM_TEST_GWP05_REAL_GGUF does not exist: ${WAM_TEST_GWP05_REAL_GGUF}")
        endif()
        add_test(NAME wam_gwp05_real_artifact
            COMMAND wam_gwp05_real_artifact
                "${WAM_TEST_GWP05_REAL_GGUF}")
        set_tests_properties(wam_gwp05_real_artifact PROPERTIES
            LABELS "integration;gwp05;artifact;external"
            TIMEOUT 120)
    else()
        message(STATUS
            "wam: skipping real GWP05 artifact gate; set WAM_TEST_GWP05_REAL_GGUF")
    endif()

    if(WAM_TEST_GWP05_ROBOTWIN_GGUF)
        if(NOT IS_ABSOLUTE "${WAM_TEST_GWP05_ROBOTWIN_GGUF}")
            message(FATAL_ERROR
                "WAM_TEST_GWP05_ROBOTWIN_GGUF must be an absolute path")
        endif()
        if(NOT EXISTS "${WAM_TEST_GWP05_ROBOTWIN_GGUF}")
            message(FATAL_ERROR
                "WAM_TEST_GWP05_ROBOTWIN_GGUF does not exist: ${WAM_TEST_GWP05_ROBOTWIN_GGUF}")
        endif()
        add_test(NAME wam_gwp05_robotwin_artifact
            COMMAND wam_gwp05_robotwin_artifact
                "${WAM_TEST_GWP05_ROBOTWIN_GGUF}")
        set_tests_properties(wam_gwp05_robotwin_artifact PROPERTIES
            LABELS "integration;gwp05;artifact;robotwin;external"
            TIMEOUT 120)
    else()
        message(STATUS
            "wam: skipping formal RoboTwin artifact gate; set WAM_TEST_GWP05_ROBOTWIN_GGUF")
    endif()

    set(wam_gwp05_robotwin_gate_variables
        WAM_TEST_GWP05_ROBOTWIN_INPUT_DIR
        WAM_TEST_GWP05_ROBOTWIN_REFERENCE_DIR)
    set(wam_gwp05_robotwin_gate_count 0)
    foreach(variable IN LISTS wam_gwp05_robotwin_gate_variables)
        if(NOT "${${variable}}" STREQUAL "")
            math(EXPR wam_gwp05_robotwin_gate_count
                "${wam_gwp05_robotwin_gate_count} + 1")
        endif()
    endforeach()
    if(wam_gwp05_robotwin_gate_count EQUAL 1)
        message(FATAL_ERROR
            "The RoboTwin input and PyTorch reference Gate A paths must be set together")
    endif()
    if(wam_gwp05_robotwin_gate_count EQUAL 2)
        if(NOT WAM_TEST_GWP05_ROBOTWIN_GGUF)
            message(FATAL_ERROR
                "RoboTwin numerical Gate A also requires WAM_TEST_GWP05_ROBOTWIN_GGUF")
        endif()
        if(NOT WAM_CUDA OR NOT WAM_CUDNN)
            message(FATAL_ERROR
                "The RoboTwin numerical Gate A requires WAM_CUDA=ON and WAM_CUDNN=ON")
        endif()
        foreach(variable IN ITEMS
                WAM_TEST_GWP05_ROBOTWIN_INPUT_DIR
                WAM_TEST_GWP05_ROBOTWIN_REFERENCE_DIR)
            if(NOT IS_ABSOLUTE "${${variable}}" OR
               NOT IS_DIRECTORY "${${variable}}")
                message(FATAL_ERROR
                    "${variable} must name an existing absolute directory")
            endif()
        endforeach()
        add_test(NAME wam_gwp05_robotwin_parity
            COMMAND wam_gwp05_robotwin_parity
                "${WAM_TEST_GWP05_ROBOTWIN_GGUF}"
                "${WAM_TEST_GWP05_ROBOTWIN_INPUT_DIR}"
                "${WAM_TEST_GWP05_ROBOTWIN_REFERENCE_DIR}")
        set_tests_properties(wam_gwp05_robotwin_parity PROPERTIES
            LABELS "integration;gwp05;cuda;bf16;robotwin;parity;external"
            TIMEOUT 900
            RUN_SERIAL TRUE)
    else()
        message(STATUS
            "wam: skipping RoboTwin BF16/CUDA parity; set all three Gate A paths")
    endif()

    if(WAM_TEST_GWP05_REAL_GGUF AND
       WAM_TEST_GWP05_MATERIALIZED_INPUT_DIR AND
       WAM_TEST_GWP05_STAGE_ROOT)
        foreach(path IN ITEMS
                WAM_TEST_GWP05_MATERIALIZED_INPUT_DIR
                WAM_TEST_GWP05_STAGE_ROOT)
            if(NOT IS_ABSOLUTE "${${path}}" OR NOT EXISTS "${${path}}")
                message(FATAL_ERROR "${path} must name an existing absolute path")
            endif()
        endforeach()
        add_test(NAME wam_gwp05_pipeline_reference
            COMMAND wam_gwp05_pipeline_reference
                "${WAM_TEST_GWP05_REAL_GGUF}"
                "${WAM_TEST_GWP05_MATERIALIZED_INPUT_DIR}"
                "${WAM_TEST_GWP05_STAGE_ROOT}")
        set_tests_properties(wam_gwp05_pipeline_reference PROPERTIES
            LABELS "integration;gwp05;pipeline;parity;external"
            TIMEOUT 900
            RUN_SERIAL TRUE)
    else()
        message(STATUS
            "wam: skipping GWP05 pipeline parity gate; set real GGUF, materialized input, and stage root")
    endif()
endif()

if(WAM_BUILD_FASTWAM)
    wam_add_test(wam_fastwam_pipeline unit/fastwam_pipeline.cpp)
    wam_add_test(wam_fastwam_semantics unit/fastwam_semantics.cpp)

    add_executable(wam_fastwam_libero_artifact
        integration/fastwam_libero_artifact.cpp)
    target_link_libraries(wam_fastwam_libero_artifact PRIVATE wam::core)
    target_include_directories(wam_fastwam_libero_artifact PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(wam_fastwam_libero_artifact PRIVATE cxx_std_17)
    target_compile_options(wam_fastwam_libero_artifact PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)

    add_executable(wam_fastwam_robotwin_artifact
        integration/fastwam_robotwin_artifact.cpp)
    target_link_libraries(wam_fastwam_robotwin_artifact PRIVATE wam::core)
    target_include_directories(wam_fastwam_robotwin_artifact PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(wam_fastwam_robotwin_artifact PRIVATE cxx_std_17)
    target_compile_options(wam_fastwam_robotwin_artifact PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)

    if(WAM_TEST_FASTWAM_LIBERO_GGUF)
        if(NOT IS_ABSOLUTE "${WAM_TEST_FASTWAM_LIBERO_GGUF}" OR
           NOT EXISTS "${WAM_TEST_FASTWAM_LIBERO_GGUF}")
            message(FATAL_ERROR
                "WAM_TEST_FASTWAM_LIBERO_GGUF must name an existing absolute file")
        endif()
        add_test(NAME wam_fastwam_libero_artifact
            COMMAND wam_fastwam_libero_artifact
                "${WAM_TEST_FASTWAM_LIBERO_GGUF}")
        set_tests_properties(wam_fastwam_libero_artifact PROPERTIES
            LABELS "integration;fastwam;artifact;libero;external"
            TIMEOUT 120)
    else()
        message(STATUS
            "wam: skipping FastWAM LIBERO artifact Gate B; set WAM_TEST_FASTWAM_LIBERO_GGUF")
    endif()

    if(WAM_TEST_FASTWAM_ROBOTWIN_GGUF)
        if(NOT IS_ABSOLUTE "${WAM_TEST_FASTWAM_ROBOTWIN_GGUF}" OR
           NOT EXISTS "${WAM_TEST_FASTWAM_ROBOTWIN_GGUF}")
            message(FATAL_ERROR
                "WAM_TEST_FASTWAM_ROBOTWIN_GGUF must name an existing absolute file")
        endif()
        add_test(NAME wam_fastwam_robotwin_artifact
            COMMAND wam_fastwam_robotwin_artifact
                "${WAM_TEST_FASTWAM_ROBOTWIN_GGUF}")
        set_tests_properties(wam_fastwam_robotwin_artifact PROPERTIES
            LABELS "integration;fastwam;artifact;robotwin;external"
            TIMEOUT 120)
    else()
        message(STATUS
            "wam: skipping FastWAM RoboTwin artifact gate; set WAM_TEST_FASTWAM_ROBOTWIN_GGUF")
    endif()

    if(WAM_CUDA AND WAM_TEST_FASTWAM_LIBERO_GGUF AND
       WAM_TEST_FASTWAM_LIBERO_REPLAY_DIR)
        if(NOT IS_ABSOLUTE "${WAM_TEST_FASTWAM_LIBERO_REPLAY_DIR}" OR
           NOT IS_DIRECTORY "${WAM_TEST_FASTWAM_LIBERO_REPLAY_DIR}")
            message(FATAL_ERROR
                "WAM_TEST_FASTWAM_LIBERO_REPLAY_DIR must name an existing absolute directory")
        endif()
        add_executable(wam_fastwam_libero_parity
            integration/fastwam_libero_parity.cpp)
        target_link_libraries(wam_fastwam_libero_parity PRIVATE wam::core)
        target_include_directories(wam_fastwam_libero_parity PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_features(wam_fastwam_libero_parity PRIVATE cxx_std_17)
        target_compile_options(wam_fastwam_libero_parity PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
        add_test(NAME wam_fastwam_libero_parity
            COMMAND wam_fastwam_libero_parity
                "${WAM_TEST_FASTWAM_LIBERO_GGUF}"
                "${WAM_TEST_FASTWAM_LIBERO_REPLAY_DIR}")
        set_tests_properties(wam_fastwam_libero_parity PROPERTIES
            LABELS "fastwam;libero;cuda;parity;external"
            RUN_SERIAL TRUE)
    endif()
endif()

target_include_directories(wam_registry_unit PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(wam_gguf_reader_unit PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(wam_artifact_layer_unit PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(wam_input_validation_unit PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(wam_policy_spec_unit PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(wam_policy_spec_failures PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(wam_policy_ops_unit PRIVATE ${PROJECT_SOURCE_DIR}/src)

if(WAM_BUILD_GWP05)
    target_include_directories(wam_gwp05_semantics PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(wam_gwp05_pipeline PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(wam_gwp05_inputs PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()

if(WAM_BUILD_FASTWAM)
    target_include_directories(wam_fastwam_pipeline PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(wam_fastwam_semantics PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
endif()

foreach(target IN ITEMS
        wam_gguf_reader_unit
        wam_artifact_layer_unit
        wam_policy_spec_unit
        wam_policy_spec_failures)
    target_link_libraries(${target} PRIVATE wam_test_fixtures)
endforeach()
if(WAM_BUILD_FASTWAM)
    target_link_libraries(wam_model_load_metadata PRIVATE wam_test_fixtures)
endif()

if(WAM_BUILD_GWP05)
    foreach(target IN ITEMS
            wam_gwp05_artifact
            wam_gwp05_pipeline
            wam_gwp05_semantics
            wam_gwp05_inputs)
        target_link_libraries(${target} PRIVATE wam_test_fixtures)
    endforeach()
endif()

target_compile_definitions(wam_registry_unit PRIVATE
    WAM_BUILD_GWP05=$<BOOL:${WAM_BUILD_GWP05}>
    WAM_BUILD_FASTWAM=$<BOOL:${WAM_BUILD_FASTWAM}>)

set(wam_gwp05_reference_variables
    WAM_TEST_GWP05_REPLAY_ROOT
    WAM_TEST_GWP05_STAGE_ROOT
    WAM_TEST_GWP05_CACHE_STAGE_ROOT
    WAM_TEST_GWP05_DONOR_EXECUTABLE)
set(wam_gwp05_reference_count 0)
foreach(variable IN LISTS wam_gwp05_reference_variables)
    if(NOT "${${variable}}" STREQUAL "")
        math(EXPR wam_gwp05_reference_count
            "${wam_gwp05_reference_count} + 1")
        if(NOT IS_ABSOLUTE "${${variable}}")
            message(FATAL_ERROR "${variable} must be an absolute path")
        endif()
        if(NOT EXISTS "${${variable}}")
            message(FATAL_ERROR "${variable} does not exist: ${${variable}}")
        endif()
    endif()
endforeach()

if(wam_gwp05_reference_count GREATER 0 AND
   NOT wam_gwp05_reference_count EQUAL 4)
    message(FATAL_ERROR
        "The four WAM_TEST_GWP05_* donor-reference paths must be set together")
endif()

if(wam_gwp05_reference_count EQUAL 4)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME wam_gwp05_donor_reference
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/gwp05_reference_manifest.py
            --replay-root "${WAM_TEST_GWP05_REPLAY_ROOT}"
            --stage-root "${WAM_TEST_GWP05_STAGE_ROOT}"
            --cache-stage-root "${WAM_TEST_GWP05_CACHE_STAGE_ROOT}"
            --donor-executable "${WAM_TEST_GWP05_DONOR_EXECUTABLE}"
            --manifest
                ${CMAKE_CURRENT_SOURCE_DIR}/reference/gwp05_donor_reference.json)
    set_tests_properties(wam_gwp05_donor_reference PROPERTIES
        LABELS "integration;gwp05;parity;external"
        TIMEOUT 120)
else()
    message(STATUS
        "wam: skipping GWP05 donor-reference gate; set all four external paths")
endif()

if(WAM_BUILD_SERVING AND TARGET wam_proto_descriptor)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME wam_rpc_protocol
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/rpc/test_protocol.py
            --descriptor ${WAM_PROTO_DESCRIPTOR})
    set_tests_properties(wam_rpc_protocol PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/eval")

    add_test(NAME wam_language_provider
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/rpc/test_language_provider.py)
    set_tests_properties(wam_language_provider PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/eval")

    add_test(NAME wam_libero_manifest
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/rpc/test_libero_manifest.py)
    set_tests_properties(wam_libero_manifest PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/eval")

    add_test(NAME wam_liberox_adapter
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/rpc/test_liberox_adapter.py)
    set_tests_properties(wam_liberox_adapter PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/eval")

    add_test(NAME wam_liberox_manifest
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/rpc/test_liberox_manifest.py)
    set_tests_properties(wam_liberox_manifest PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/eval")
endif()
