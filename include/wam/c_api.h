#pragma once

#include <stddef.h>
#include <stdint.h>

#define WAM_C_ABI_VERSION 3U

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

typedef struct wam_c_model_options {
    size_t struct_size;
    const char * artifact_path;
    uint32_t backend;
    uint32_t compute_precision;
    int32_t device_index;
    size_t prompt_cache_capacity;
    uint32_t language_mode;
} wam_c_model_options;

typedef struct wam_c_session_options {
    size_t struct_size;
    uint8_t enable_prefix_cache;
    uint64_t random_seed;
} wam_c_session_options;

typedef struct wam_c_image {
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
    const void * data;
    size_t byte_size;
    uint32_t dtype;
    const int64_t * shape;
    size_t rank;
    const char * layout;
    uint32_t byte_order;
} wam_c_tensor_view;

typedef struct wam_c_predict_inputs {
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
    char * name;
    double milliseconds;
} wam_c_phase_timing;

typedef struct wam_c_stats {
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
    uint8_t * action_data;
    size_t action_byte_size;
    uint32_t action_dtype;
    int64_t * action_shape;
    size_t action_rank;
    wam_c_stats stats;
} wam_c_prediction;

typedef struct wam_c_error {
    uint32_t code;
    char * message;
    char * details_json;
} wam_c_error;

WAM_C_API uint32_t wam_c_abi_version(void);
WAM_C_API int wam_c_model_create(const wam_c_model_options * options,
                                 wam_c_model ** model,
                                 wam_c_error * error);
WAM_C_API int wam_c_model_metadata_json(const wam_c_model * model,
                                        char ** metadata_json,
                                        wam_c_error * error);
WAM_C_API int wam_c_session_create(wam_c_model * model,
                                   const wam_c_session_options * options,
                                   wam_c_session ** session,
                                   wam_c_error * error);
WAM_C_API int wam_c_session_predict(wam_c_session * session,
                                    const wam_c_predict_inputs * inputs,
                                    wam_c_prediction * prediction,
                                    wam_c_error * error);
WAM_C_API int wam_c_session_reset(wam_c_session * session,
                                  wam_c_error * error);
WAM_C_API void wam_c_prediction_free(wam_c_prediction * prediction);
WAM_C_API void wam_c_string_free(char * value);
WAM_C_API void wam_c_error_free(wam_c_error * error);
WAM_C_API void wam_c_session_free(wam_c_session * session);
WAM_C_API void wam_c_model_free(wam_c_model * model);

#ifdef __cplusplus
}
#endif
