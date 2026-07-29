#pragma once

#include <stddef.h>
#include <stdint.h>

#define WAM_C_ABI_VERSION 4U
#define WAM_C_STRUCT_VERSION 1U

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

typedef struct wam_c_model wam_c_model;
typedef struct wam_c_session wam_c_session;

typedef enum wam_c_backend {
    WAM_C_BACKEND_UNKNOWN = 0,
    WAM_C_BACKEND_AUTOMATIC = 1,
    WAM_C_BACKEND_CUDA = 2,
    WAM_C_BACKEND_CPU_METADATA = 3,
    WAM_C_BACKEND_CPU = 4,
} wam_c_backend;

typedef enum wam_c_compute_precision {
    WAM_C_PRECISION_UNKNOWN = 0,
    WAM_C_PRECISION_AUTOMATIC = 1,
    WAM_C_PRECISION_F32 = 2,
    WAM_C_PRECISION_F16 = 3,
    WAM_C_PRECISION_BF16 = 4,
    WAM_C_PRECISION_FP8_E4M3 = 5,
    WAM_C_PRECISION_FP8_E5M2 = 6,
    WAM_C_PRECISION_INT8 = 7,
} wam_c_compute_precision;

typedef enum wam_c_language_mode {
    WAM_C_LANGUAGE_AUTOMATIC = 0,
    WAM_C_LANGUAGE_TOKENS = 1,
    WAM_C_LANGUAGE_EXTERNAL_EMBEDDING = 2,
} wam_c_language_mode;

typedef enum wam_c_status {
    WAM_C_STATUS_OK = 0,
    WAM_C_STATUS_ERROR = 1,
} wam_c_status;

typedef struct wam_c_model_options {
    uint32_t struct_version;
    size_t struct_size;
    const char * artifact_path;
    uint32_t backend;
    uint32_t compute_precision;
    int32_t device_index;
    size_t prompt_cache_capacity;
    uint32_t language_mode;
} wam_c_model_options;

typedef struct wam_c_session_options {
    uint32_t struct_version;
    size_t struct_size;
    uint8_t enable_prefix_cache;
    uint64_t random_seed;
} wam_c_session_options;

typedef struct wam_c_image {
    uint32_t struct_version;
    size_t struct_size;
    const char * name;
    uint32_t encoding;
    const uint8_t * data;
    size_t byte_size;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    size_t row_stride_bytes;
} wam_c_image;

typedef struct wam_c_tensor_view {
    uint32_t struct_version;
    size_t struct_size;
    const void * data;
    size_t byte_size;
    uint32_t dtype;
    const int64_t * shape;
    size_t rank;
    const char * layout;
    uint32_t byte_order;
} wam_c_tensor_view;

typedef struct wam_c_predict_inputs {
    uint32_t struct_version;
    size_t struct_size;
    const wam_c_image * images;
    size_t image_count;
    const int32_t * token_ids;
    const int32_t * attention_mask;
    size_t token_count;
    wam_c_tensor_view state;
    wam_c_tensor_view action_noise;
    wam_c_tensor_view embedding;
    wam_c_tensor_view embedding_attention_mask;
} wam_c_predict_inputs;

typedef struct wam_c_phase_timing {
    uint32_t struct_version;
    size_t struct_size;
    char * name;
    double milliseconds;
} wam_c_phase_timing;

typedef struct wam_c_stats {
    uint32_t struct_version;
    size_t struct_size;
    double preprocess_milliseconds;
    double model_milliseconds;
    double model_vision_milliseconds;
    double model_text_milliseconds;
    double model_prefill_milliseconds;
    double model_decode_milliseconds;
    double postprocess_milliseconds;
    double total_milliseconds;
    uint64_t peak_device_memory_bytes;
    wam_c_phase_timing * model_timings;
    size_t model_timing_count;
} wam_c_stats;

typedef struct wam_c_prediction {
    uint32_t struct_version;
    size_t struct_size;
    uint8_t * action_data;
    size_t action_byte_size;
    uint32_t action_dtype;
    int64_t * action_shape;
    size_t action_rank;
    wam_c_stats stats;
} wam_c_prediction;

typedef struct wam_c_error {
    uint32_t struct_version;
    size_t struct_size;
    uint32_t code;
    char * message;
    char * details_json;
} wam_c_error;

WAM_C_API uint32_t wam_c_abi_version(void);

/*
 * Initializers zero the complete struct, set its current version/size, and
 * apply defaults. Callers must use them rather than aggregate defaults.
 * A future larger struct is accepted when its leading v1 layout is unchanged.
 */
WAM_C_API void wam_c_model_options_init(wam_c_model_options * options);
WAM_C_API void wam_c_session_options_init(wam_c_session_options * options);
WAM_C_API void wam_c_image_init(wam_c_image * image);
WAM_C_API void wam_c_tensor_view_init(wam_c_tensor_view * tensor);
WAM_C_API void wam_c_predict_inputs_init(wam_c_predict_inputs * inputs);

/*
 * Input structs and their buffers are borrowed until the call returns.
 * Output parameters are set to NULL before work starts. On success, returned
 * handles, strings, and predictions are owned by the caller. On failure, an
 * optional non-NULL error output is owned by the caller. A Session retains the
 * model resources it needs and remains valid after its Model handle is freed.
 */
WAM_C_API int wam_c_model_create(const wam_c_model_options * options,
                                 wam_c_model ** model,
                                 wam_c_error ** error);
WAM_C_API int wam_c_model_metadata_json(const wam_c_model * model,
                                        char ** metadata_json,
                                        wam_c_error ** error);
WAM_C_API int wam_c_session_create(wam_c_model * model,
                                   const wam_c_session_options * options,
                                   wam_c_session ** session,
                                   wam_c_error ** error);
WAM_C_API int wam_c_session_predict(wam_c_session * session,
                                    const wam_c_predict_inputs * inputs,
                                    wam_c_prediction ** prediction,
                                    wam_c_error ** error);
WAM_C_API int wam_c_session_reset(wam_c_session * session,
                                  wam_c_error ** error);

/* Every free function accepts NULL. Do not use another allocator to free data. */
WAM_C_API void wam_c_prediction_free(wam_c_prediction * prediction);
WAM_C_API void wam_c_string_free(char * value);
WAM_C_API void wam_c_error_free(wam_c_error * error);
WAM_C_API void wam_c_session_free(wam_c_session * session);
WAM_C_API void wam_c_model_free(wam_c_model * model);

#ifdef __cplusplus
}
#endif
