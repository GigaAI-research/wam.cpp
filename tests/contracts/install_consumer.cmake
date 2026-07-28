if(NOT DEFINED WAM_BINARY_DIR OR NOT DEFINED WAM_CONSUMER_SOURCE_DIR)
    message(FATAL_ERROR "WAM_BINARY_DIR and WAM_CONSUMER_SOURCE_DIR are required")
endif()

set(work_dir "${WAM_BINARY_DIR}/tests/install-consumer")
set(prefix_dir "${work_dir}/prefix")
set(consumer_build_dir "${work_dir}/build")
file(REMOVE_RECURSE "${work_dir}")

set(config_args "")
if(DEFINED WAM_TEST_CONFIG AND NOT WAM_TEST_CONFIG STREQUAL "")
    list(APPEND config_args --config "${WAM_TEST_CONFIG}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${WAM_BINARY_DIR}"
        --prefix "${prefix_dir}" ${config_args}
    RESULT_VARIABLE install_result)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "wam install failed: ${install_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${WAM_CONSUMER_SOURCE_DIR}"
        -B "${consumer_build_dir}"
        "-DCMAKE_PREFIX_PATH=${prefix_dir}"
    RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "external consumer configure failed: ${configure_result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}" ${config_args}
    RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "external consumer build failed: ${build_result}")
endif()

set(ctest_config_args "")
if(DEFINED WAM_TEST_CONFIG AND NOT WAM_TEST_CONFIG STREQUAL "")
    list(APPEND ctest_config_args -C "${WAM_TEST_CONFIG}")
endif()
execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}"
        --test-dir "${consumer_build_dir}"
        --output-on-failure
        ${ctest_config_args}
    RESULT_VARIABLE run_result)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "external consumer failed: ${run_result}")
endif()
