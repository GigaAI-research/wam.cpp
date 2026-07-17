#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(WAM_C_API_BUILD)
#    define WAM_C_API __declspec(dllexport)
#  else
#    define WAM_C_API __declspec(dllimport)
#  endif
#else
#  define WAM_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wam_rpc_service wam_rpc_service;

typedef struct wam_rpc_model_options {
    const char * artifact_path;
    uint32_t backend;
    uint32_t precision;
    int32_t device_index;
    size_t prompt_cache_capacity;
} wam_rpc_model_options;

typedef struct wam_rpc_model_options_v2 {
    size_t struct_size;
    const char * artifact_path;
    uint32_t backend;
    uint32_t precision;
    int32_t device_index;
    size_t prompt_cache_capacity;
    uint32_t language_encoder_policy;
    const int32_t * fixed_token_ids;
    size_t fixed_token_count;
    const int32_t * fixed_attention_mask;
    size_t fixed_attention_mask_count;
} wam_rpc_model_options_v2;

typedef struct wam_rpc_result {
    uint32_t error_code;
    uint32_t grpc_status_code;
    uint8_t * payload;
    size_t payload_size;
    char * message;
} wam_rpc_result;

enum wam_rpc_method {
    WAM_RPC_GET_MODEL_INFO = 1,
    WAM_RPC_CREATE_SESSION = 2,
    WAM_RPC_PREDICT = 3,
    WAM_RPC_RESET_SESSION = 4,
    WAM_RPC_CLOSE_SESSION = 5,
};

WAM_C_API uint32_t wam_rpc_c_abi_version(void);
WAM_C_API int wam_rpc_service_create(const wam_rpc_model_options * options,
                                     wam_rpc_service ** service,
                                     wam_rpc_result * result);
WAM_C_API int wam_rpc_service_create_v2(const wam_rpc_model_options_v2 * options,
                                        wam_rpc_service ** service,
                                        wam_rpc_result * result);
WAM_C_API int wam_rpc_service_call(wam_rpc_service * service,
                                   uint32_t method,
                                   const uint8_t * request,
                                   size_t request_size,
                                   wam_rpc_result * result);
WAM_C_API void wam_rpc_result_free(wam_rpc_result * result);
WAM_C_API void wam_rpc_service_free(wam_rpc_service * service);

#ifdef __cplusplus
}
#endif
