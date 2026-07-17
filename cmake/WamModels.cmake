include_guard(GLOBAL)

function(wam_add_model_sources target)
    target_sources(${target} PRIVATE
        src/models/common/gguf_reader.cpp
        src/models/common/input_validation.cpp
        src/models/common/scheduler.cpp)
    if(WAM_BUILD_GWP05)
        target_sources(${target} PRIVATE
            src/models/gwp05/model.cpp
            src/models/gwp05/inputs.cpp
            src/models/gwp05/engine/engine.cpp
            src/models/gwp05/semantics.cpp
            src/models/gwp05/artifact.cpp)
        target_include_directories(${target} PRIVATE ${llama_SOURCE_DIR}/vendor/stb)
        target_compile_definitions(${target} PRIVATE WAM_BUILD_GWP05=1)
        if(WAM_CUDA)
            target_sources(${target} PRIVATE src/models/gwp05/kernels/cuda_preprocess.cu)
            target_link_libraries(${target} PRIVATE CUDA::cudart)
            target_compile_definitions(${target} PRIVATE GGML_USE_CUDA=1)
            target_compile_definitions(${target} PRIVATE WAM_GWP05_CUDA_PREPROCESS=1)
        endif()
        if(WAM_CUDNN)
            target_compile_definitions(${target} PRIVATE WAM_GWP05_CUDNN=1)
        endif()
    endif()
endfunction()
