file(REMOVE_RECURSE "${WAM_CLI_TEST_ROOT}")
file(MAKE_DIRECTORY "${WAM_CLI_TEST_ROOT}/bundle")

set(model "${WAM_CLI_TEST_ROOT}/model.gguf")
execute_process(
    COMMAND "${WAM_FIXTURE_WRITER}" "${model}"
    RESULT_VARIABLE fixture_result
    ERROR_VARIABLE fixture_error)
if(NOT fixture_result EQUAL 0)
    message(FATAL_ERROR "fixture creation failed: ${fixture_error}")
endif()

execute_process(
    COMMAND "${WAM_INSPECT}" "${model}"
    RESULT_VARIABLE inspect_result
    OUTPUT_VARIABLE inspect_output
    ERROR_VARIABLE inspect_error)
if(NOT inspect_result EQUAL 0 OR
   NOT inspect_output MATCHES "\"valid\":true" OR
   NOT inspect_output MATCHES "\"schema_version\":3" OR
   NOT inspect_output MATCHES "\"image_roles\":\\[" OR
   NOT inspect_output MATCHES "\"state_fields\":\\[" OR
   NOT inspect_output MATCHES "\"action_fields\":\\[")
    message(FATAL_ERROR
        "direct artifact inspection failed: ${inspect_error}${inspect_output}")
endif()

file(COPY_FILE "${model}" "${WAM_CLI_TEST_ROOT}/bundle/model.gguf")
file(WRITE "${WAM_CLI_TEST_ROOT}/bundle/tokenizer.json" "{}\n")
file(WRITE "${WAM_CLI_TEST_ROOT}/bundle/manifest.json"
    "{\"format\":\"wam-bundle-v1\",\"manifest_schema_version\":1,\"model\":\"model.gguf\",\"tokenizer\":\"tokenizer.json\"}\n")
execute_process(
    COMMAND "${WAM_INSPECT}" "${WAM_CLI_TEST_ROOT}/bundle"
    RESULT_VARIABLE bundle_result
    OUTPUT_VARIABLE bundle_output
    ERROR_VARIABLE bundle_error)
if(NOT bundle_result EQUAL 0 OR
   NOT bundle_output MATCHES "\"bundle\":true" OR
   NOT bundle_output MATCHES "\"tokenizer\":\"")
    message(FATAL_ERROR
        "bundle inspection failed: ${bundle_error}${bundle_output}")
endif()

execute_process(
    COMMAND "${WAM_VALIDATE}" "${WAM_CLI_TEST_ROOT}/bundle"
    RESULT_VARIABLE validate_result
    OUTPUT_VARIABLE validate_output
    ERROR_VARIABLE validate_error)
if(WAM_EXPECT_VALIDATE)
    if(NOT validate_result EQUAL 0 OR
       NOT validate_output MATCHES "\"valid\":true" OR
       NOT validate_output MATCHES "\"architecture\":\"gwp05\"")
        message(FATAL_ERROR
            "bundle validation failed: ${validate_error}${validate_output}")
    endif()
elseif(validate_result EQUAL 0 OR
       NOT validate_error MATCHES "\"valid\":false" OR
       NOT validate_error MATCHES "\"code\":\"unsupported\"")
    message(FATAL_ERROR
        "disabled architecture was not reported: ${validate_error}${validate_output}")
endif()

file(WRITE "${WAM_CLI_TEST_ROOT}/broken.gguf" "not a GGUF")
execute_process(
    COMMAND "${WAM_INSPECT}" "${WAM_CLI_TEST_ROOT}/broken.gguf"
    RESULT_VARIABLE broken_result
    OUTPUT_VARIABLE broken_output
    ERROR_VARIABLE broken_error)
if(broken_result EQUAL 0 OR
   NOT broken_error MATCHES "\"valid\":false" OR
   NOT broken_error MATCHES "\"error\":")
    message(FATAL_ERROR
        "corrupt artifact did not produce a structured error: ${broken_output}${broken_error}")
endif()
