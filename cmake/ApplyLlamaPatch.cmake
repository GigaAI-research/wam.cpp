if(NOT DEFINED LLAMA_SOURCE_DIR OR NOT DEFINED LLAMA_PATCH_FILE OR
   NOT DEFINED PATCH_EXECUTABLE OR NOT DEFINED LLAMA_PATCH_MARKER)
    message(FATAL_ERROR
        "LLAMA_SOURCE_DIR, LLAMA_PATCH_FILE, LLAMA_PATCH_MARKER, and PATCH_EXECUTABLE are required")
endif()

set(marker_file "${LLAMA_SOURCE_DIR}/ggml/src/ggml-cuda/ggml-cuda.cu")
if(DEFINED LLAMA_PATCH_MARKER_FILE)
    set(marker_file "${LLAMA_SOURCE_DIR}/${LLAMA_PATCH_MARKER_FILE}")
endif()
if(NOT EXISTS "${marker_file}")
    message(FATAL_ERROR "llama.cpp patch marker file is missing: ${marker_file}")
endif()
file(READ "${marker_file}" marker_contents)
string(FIND "${marker_contents}" "${LLAMA_PATCH_MARKER}" marker_position)
if(NOT marker_position EQUAL -1)
    return()
endif()

execute_process(
    COMMAND "${PATCH_EXECUTABLE}" --batch --forward -p1 -i "${LLAMA_PATCH_FILE}"
    WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
    RESULT_VARIABLE patch_result
    OUTPUT_VARIABLE patch_output
    ERROR_VARIABLE patch_error)
if(NOT patch_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to apply llama patch ${LLAMA_PATCH_FILE}:\n${patch_output}${patch_error}")
endif()
