#include "wam/c_api.h"

#include "bindings/metadata_codec.h"
#include "wam/wam.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

struct wam_c_model {
    std::unique_ptr<wam::Model> value;
};

struct wam_c_session {
    std::unique_ptr<wam::Session> value;
};

namespace {

template <typename T>
void initialize(T * value) {
    if (value == nullptr) return;
    std::memset(value, 0, sizeof(*value));
    value->struct_version = WAM_C_STRUCT_VERSION;
    value->struct_size = sizeof(*value);
}

char * copy_string(const std::string & value) {
    char * result = static_cast<char *>(std::malloc(value.size() + 1));
    if (result == nullptr) throw std::bad_alloc();
    std::memcpy(result, value.c_str(), value.size() + 1);
    return result;
}

void set_error(wam_c_error ** target, wam::ErrorCode code,
               const std::string & message,
               const std::vector<wam::ErrorDetail> & details = {}) noexcept {
    if (target == nullptr) return;
    wam_c_error * result = static_cast<wam_c_error *>(
        std::calloc(1, sizeof(wam_c_error)));
    if (result == nullptr) return;
    initialize(result);
    result->code = static_cast<std::uint32_t>(code);
    try {
        result->message = copy_string(message);
        result->details_json = copy_string(
            wam::bindings::error_details_json(details));
    } catch (...) {
        std::free(result->message);
        std::free(result->details_json);
        result->message = nullptr;
        result->details_json = nullptr;
        result->code = static_cast<std::uint32_t>(wam::ErrorCode::internal);
    }
    *target = result;
}

template <typename Function>
int guarded(wam_c_error ** error, Function && function) noexcept {
    if (error != nullptr) *error = nullptr;
    try {
        function();
        return WAM_C_STATUS_OK;
    } catch (const wam::Error & failure) {
        set_error(error, failure.code(), failure.what(), failure.details());
    } catch (const std::bad_alloc &) {
        set_error(error, wam::ErrorCode::resource_exhausted,
                  "C ABI allocation failed");
    } catch (const std::exception & failure) {
        set_error(error, wam::ErrorCode::internal, failure.what());
    } catch (...) {
        set_error(error, wam::ErrorCode::internal,
                  "unknown C ABI failure");
    }
    return WAM_C_STATUS_ERROR;
}

void require(bool condition, const char * message, const char * field) {
    if (!condition) {
        throw wam::Error(wam::ErrorCode::invalid_argument, message,
                         {{field, "invalid C ABI value"}});
    }
}

template <typename T>
void require_struct(const T * value, const char * field) {
    require(value != nullptr, "C ABI struct pointer is null", field);
    require(value->struct_version == WAM_C_STRUCT_VERSION,
            "C ABI struct version is unsupported", field);
    require(value->struct_size >= sizeof(T),
            "C ABI struct size is too small", field);
}

wam::TensorView tensor_view(const wam_c_tensor_view & source,
                            const char * field) {
    require_struct(&source, field);
    require(source.rank == 0 || source.shape != nullptr,
            "tensor shape is null", field);
    std::vector<std::int64_t> shape;
    if (source.rank != 0) {
        shape.assign(source.shape, source.shape + source.rank);
    }
    return {source.data, source.byte_size,
            static_cast<wam::DType>(source.dtype), std::move(shape),
            source.layout == nullptr ? "" : source.layout,
            static_cast<wam::ByteOrder>(source.byte_order)};
}

wam::Observation runtime_inputs(const wam_c_predict_inputs & source) {
    require_struct(&source, "inputs");
    require(source.image_count == 0 || source.images != nullptr,
            "images pointer is null", "inputs.images");
    require(source.token_count == 0 ||
                (source.token_ids != nullptr && source.attention_mask != nullptr),
            "token pointers are null", "inputs.language");
    wam::Observation result;
    result.images.reserve(source.image_count);
    for (std::size_t index = 0; index < source.image_count; ++index) {
        const wam_c_image & image = source.images[index];
        require_struct(&image, "inputs.images");
        require(image.name != nullptr, "image name is null",
                "inputs.images.name");
        result.images.push_back({
            image.name, static_cast<wam::ImageEncoding>(image.encoding),
            image.data, image.byte_size, image.width, image.height,
            image.channels, image.row_stride_bytes});
    }
    const bool has_tokens = source.token_count != 0;
    const bool has_embedding = source.embedding.data != nullptr ||
                               source.embedding.byte_size != 0 ||
                               source.embedding.rank != 0;
    require(!(has_tokens && has_embedding),
            "token and embedding language inputs are mutually exclusive",
            "inputs.language");
    if (has_tokens) {
        result.language = wam::TokenInput{
            wam::ArrayView<std::int32_t>(source.token_ids, source.token_count),
            wam::ArrayView<std::int32_t>(source.attention_mask,
                                         source.token_count)};
    } else if (has_embedding) {
        result.language = wam::EmbeddingInput{
            tensor_view(source.embedding, "inputs.language.embedding"),
            tensor_view(source.embedding_attention_mask,
                        "inputs.language.attention_mask")};
    }
    result.state = tensor_view(source.state, "inputs.state");
    result.action_noise = tensor_view(source.action_noise,
                                      "inputs.action_noise");
    return result;
}

wam_c_prediction * copy_prediction(const wam::Prediction & source) {
    auto * target = static_cast<wam_c_prediction *>(
        std::calloc(1, sizeof(wam_c_prediction)));
    if (target == nullptr) throw std::bad_alloc();
    initialize(target);
    initialize(&target->stats);
    try {
        target->action_byte_size = source.action.data.size();
        if (!source.action.data.empty()) {
            target->action_data = static_cast<std::uint8_t *>(
                std::malloc(source.action.data.size()));
            if (target->action_data == nullptr) throw std::bad_alloc();
            std::memcpy(target->action_data, source.action.data.data(),
                        source.action.data.size());
        }
        target->action_dtype = static_cast<std::uint32_t>(source.action.dtype);
        target->action_rank = source.action.shape.size();
        if (!source.action.shape.empty()) {
            target->action_shape = static_cast<std::int64_t *>(
                std::malloc(source.action.shape.size() * sizeof(std::int64_t)));
            if (target->action_shape == nullptr) throw std::bad_alloc();
            std::memcpy(target->action_shape, source.action.shape.data(),
                        source.action.shape.size() * sizeof(std::int64_t));
        }
        const wam::Telemetry & stats = source.telemetry;
        target->stats.preprocess_milliseconds = stats.preprocess_milliseconds;
        target->stats.model_milliseconds = stats.model_milliseconds;
        target->stats.model_vision_milliseconds =
            stats.model_vision_milliseconds;
        target->stats.model_text_milliseconds = stats.model_text_milliseconds;
        target->stats.model_prefill_milliseconds =
            stats.model_prefill_milliseconds;
        target->stats.model_decode_milliseconds =
            stats.model_decode_milliseconds;
        target->stats.postprocess_milliseconds = stats.postprocess_milliseconds;
        target->stats.total_milliseconds = stats.total_milliseconds;
        target->stats.peak_device_memory_bytes =
            stats.peak_device_memory_bytes;
        if (!stats.model_timings.empty()) {
            target->stats.model_timings = static_cast<wam_c_phase_timing *>(
                std::calloc(stats.model_timings.size(),
                            sizeof(wam_c_phase_timing)));
            if (target->stats.model_timings == nullptr) throw std::bad_alloc();
            target->stats.model_timing_count = stats.model_timings.size();
            for (std::size_t index = 0; index < stats.model_timings.size();
                 ++index) {
                initialize(&target->stats.model_timings[index]);
                target->stats.model_timings[index].name =
                    copy_string(stats.model_timings[index].name);
                target->stats.model_timings[index].milliseconds =
                    stats.model_timings[index].milliseconds;
            }
        }
        return target;
    } catch (...) {
        wam_c_prediction_free(target);
        throw;
    }
}

} // namespace

extern "C" {

std::uint32_t wam_c_abi_version(void) { return WAM_C_ABI_VERSION; }

void wam_c_model_options_init(wam_c_model_options * options) {
    initialize(options);
    if (options == nullptr) return;
    options->backend = WAM_C_BACKEND_AUTOMATIC;
    options->compute_precision = WAM_C_PRECISION_AUTOMATIC;
    options->language_mode = WAM_C_LANGUAGE_AUTOMATIC;
}

void wam_c_session_options_init(wam_c_session_options * options) {
    initialize(options);
    if (options != nullptr) options->enable_prefix_cache = 1;
}

void wam_c_image_init(wam_c_image * image) { initialize(image); }

void wam_c_tensor_view_init(wam_c_tensor_view * tensor) {
    initialize(tensor);
    if (tensor != nullptr) tensor->byte_order = 3;
}

void wam_c_predict_inputs_init(wam_c_predict_inputs * inputs) {
    initialize(inputs);
    if (inputs == nullptr) return;
    wam_c_tensor_view_init(&inputs->state);
    wam_c_tensor_view_init(&inputs->action_noise);
    wam_c_tensor_view_init(&inputs->embedding);
    wam_c_tensor_view_init(&inputs->embedding_attention_mask);
}

int wam_c_model_create(const wam_c_model_options * options,
                       wam_c_model ** model, wam_c_error ** error) {
    if (model != nullptr) *model = nullptr;
    return guarded(error, [&] {
        require_struct(options, "options");
        require(model != nullptr, "model output is null", "model");
        require(options->artifact_path != nullptr,
                "artifact path is null", "options.artifact_path");
        wam::RuntimeConfig runtime;
        runtime.backend = static_cast<wam::Backend>(options->backend);
        runtime.compute_precision =
            static_cast<wam::ComputePrecision>(options->compute_precision);
        runtime.device_index = options->device_index;
        runtime.prompt_cache_capacity = options->prompt_cache_capacity;
        runtime.language_mode =
            static_cast<wam::LanguageRuntimeMode>(options->language_mode);
        auto holder = std::make_unique<wam_c_model>();
        holder->value = std::make_unique<wam::Model>(
            wam::Model::load(options->artifact_path, runtime));
        *model = holder.release();
    });
}

int wam_c_model_metadata_json(const wam_c_model * model, char ** metadata_json,
                              wam_c_error ** error) {
    if (metadata_json != nullptr) *metadata_json = nullptr;
    return guarded(error, [&] {
        require(model != nullptr && model->value != nullptr,
                "model is null", "model");
        require(metadata_json != nullptr, "metadata output is null",
                "metadata_json");
        const wam::ModelInfo & info = model->value->info();
        require(info.policy_spec != nullptr,
                "model PolicySpec is unavailable", "model.policy_spec");
        *metadata_json = copy_string(
            wam::bindings::model_metadata_json(info, *info.policy_spec));
    });
}

int wam_c_session_create(wam_c_model * model,
                         const wam_c_session_options * options,
                         wam_c_session ** session, wam_c_error ** error) {
    if (session != nullptr) *session = nullptr;
    return guarded(error, [&] {
        require(model != nullptr && model->value != nullptr,
                "model is null", "model");
        require_struct(options, "options");
        require(session != nullptr, "session output is null", "session");
        auto holder = std::make_unique<wam_c_session>();
        holder->value = std::make_unique<wam::Session>(
            model->value->create_session(
                {options->enable_prefix_cache != 0, options->random_seed}));
        *session = holder.release();
    });
}

int wam_c_session_predict(wam_c_session * session,
                          const wam_c_predict_inputs * inputs,
                          wam_c_prediction ** prediction,
                          wam_c_error ** error) {
    if (prediction != nullptr) *prediction = nullptr;
    return guarded(error, [&] {
        require(session != nullptr && session->value != nullptr,
                "session is null", "session");
        require_struct(inputs, "inputs");
        require(prediction != nullptr, "prediction output is null", "prediction");
        wam::Observation runtime = runtime_inputs(*inputs);
        *prediction = copy_prediction(session->value->predict(runtime));
    });
}

int wam_c_session_reset(wam_c_session * session, wam_c_error ** error) {
    return guarded(error, [&] {
        require(session != nullptr && session->value != nullptr,
                "session is null", "session");
        session->value->reset();
    });
}

void wam_c_prediction_free(wam_c_prediction * prediction) {
    if (prediction == nullptr) return;
    std::free(prediction->action_data);
    std::free(prediction->action_shape);
    for (std::size_t index = 0;
         index < prediction->stats.model_timing_count; ++index) {
        std::free(prediction->stats.model_timings[index].name);
    }
    std::free(prediction->stats.model_timings);
    std::free(prediction);
}

void wam_c_string_free(char * value) { std::free(value); }

void wam_c_error_free(wam_c_error * error) {
    if (error == nullptr) return;
    std::free(error->message);
    std::free(error->details_json);
    std::free(error);
}

void wam_c_session_free(wam_c_session * session) { delete session; }

void wam_c_model_free(wam_c_model * model) { delete model; }

} // extern "C"
