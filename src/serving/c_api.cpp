#include "wam/c_api.h"

#include "model_internal.h"
#include "serving/protocol_adapter.h"
#include "wam/wam.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <vector>

struct wam_c_model {
    wam::Model * value = nullptr;
};

struct wam_c_session {
    wam::Session * value = nullptr;
};

namespace {

void clear_error(wam_c_error * error) {
    if (error == nullptr) return;
    error->code = 0;
    error->message = nullptr;
    error->details_json = nullptr;
}

char * copy_string(const std::string & value) {
    char * result = static_cast<char *>(std::malloc(value.size() + 1));
    if (result == nullptr) throw std::bad_alloc();
    std::memcpy(result, value.c_str(), value.size() + 1);
    return result;
}

void set_error(wam_c_error * target, wam::ErrorCode code,
               const std::string & message,
               const std::vector<wam::ErrorDetail> & details = {}) noexcept {
    if (target == nullptr) return;
    try {
        target->code = static_cast<std::uint32_t>(code);
        target->message = copy_string(message);
        target->details_json = copy_string(wam::serving::error_details_json(details));
    } catch (...) {
        std::free(target->message);
        std::free(target->details_json);
        target->code = static_cast<std::uint32_t>(wam::ErrorCode::internal);
        target->message = nullptr;
        target->details_json = nullptr;
    }
}

template <typename Function>
int guarded(wam_c_error * error, Function && function) noexcept {
    clear_error(error);
    try {
        function();
        return 0;
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
    return 1;
}

void require(bool condition, const char * message, const char * field) {
    if (!condition) {
        throw wam::Error(wam::ErrorCode::invalid_argument, message,
                         {{field, "invalid C ABI value"}});
    }
}

wam::TensorView tensor_view(const wam_c_tensor_view & source,
                            const char * field) {
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

wam::Inputs runtime_inputs(const wam_c_predict_inputs & source) {
    require(source.image_count == 0 || source.images != nullptr,
            "images pointer is null", "inputs.images");
    require(source.token_count == 0 ||
                (source.token_ids != nullptr && source.attention_mask != nullptr),
            "token pointers are null", "inputs.language");
    wam::Inputs result;
    result.images.reserve(source.image_count);
    for (std::size_t index = 0; index < source.image_count; ++index) {
        const wam_c_image & image = source.images[index];
        require(image.name != nullptr, "image name is null", "inputs.images.name");
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

void copy_prediction(const wam::Prediction & source,
                     wam_c_prediction * target) {
    require(target != nullptr, "prediction output is null", "prediction");
    std::memset(target, 0, sizeof(*target));
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
        if (target->action_shape == nullptr) {
            std::free(target->action_data);
            std::memset(target, 0, sizeof(*target));
            throw std::bad_alloc();
        }
        std::memcpy(target->action_shape, source.action.shape.data(),
                    source.action.shape.size() * sizeof(std::int64_t));
    }
    const wam::Stats & stats = source.stats;
    target->stats = {
        stats.preprocess_milliseconds, stats.model_milliseconds,
        stats.model_vision_milliseconds, stats.model_text_milliseconds,
        stats.model_prefill_milliseconds, stats.model_decode_milliseconds,
        stats.postprocess_milliseconds, stats.total_milliseconds,
        stats.peak_device_memory_bytes, nullptr, 0};
    if (!stats.model_timings.empty()) {
        target->stats.model_timings = static_cast<wam_c_phase_timing *>(
            std::calloc(stats.model_timings.size(), sizeof(wam_c_phase_timing)));
        if (target->stats.model_timings == nullptr) {
            wam_c_prediction_free(target);
            throw std::bad_alloc();
        }
        target->stats.model_timing_count = stats.model_timings.size();
        try {
            for (std::size_t index = 0; index < stats.model_timings.size();
                 ++index) {
                target->stats.model_timings[index].name =
                    copy_string(stats.model_timings[index].name);
                target->stats.model_timings[index].milliseconds =
                    stats.model_timings[index].milliseconds;
            }
        } catch (...) {
            wam_c_prediction_free(target);
            throw;
        }
    }
}

} // namespace

extern "C" {

std::uint32_t wam_c_abi_version(void) { return WAM_C_ABI_VERSION; }

int wam_c_model_create(const wam_c_model_options * options,
                       wam_c_model ** model, wam_c_error * error) {
    return guarded(error, [&] {
        require(options != nullptr, "model options are null", "options");
        require(model != nullptr, "model output is null", "model");
        *model = nullptr;
        require(options->struct_size == sizeof(wam_c_model_options),
                "model options size does not match ABI", "options.struct_size");
        require(options->artifact_path != nullptr,
                "artifact path is null", "options.artifact_path");
        wam::ModelOptions runtime;
        runtime.artifact_path = options->artifact_path;
        runtime.backend = static_cast<wam::Backend>(options->backend);
        runtime.compute_precision =
            static_cast<wam::ComputePrecision>(options->compute_precision);
        runtime.device_index = options->device_index;
        runtime.prompt_cache_capacity = options->prompt_cache_capacity;
        runtime.language_mode =
            static_cast<wam::LanguageRuntimeMode>(options->language_mode);
        auto holder = new wam_c_model();
        try {
            holder->value = wam::model_load(runtime);
        } catch (...) {
            delete holder;
            throw;
        }
        *model = holder;
    });
}

int wam_c_model_metadata_json(const wam_c_model * model, char ** metadata_json,
                              wam_c_error * error) {
    return guarded(error, [&] {
        require(model != nullptr && model->value != nullptr,
                "model is null", "model");
        require(metadata_json != nullptr, "metadata output is null",
                "metadata_json");
        *metadata_json = nullptr;
        const wam::internal::ModelImpl & impl =
            wam::internal::model_impl(model->value);
        *metadata_json = copy_string(wam::serving::model_metadata_json(
            impl.info(), impl.policy_spec()));
    });
}

int wam_c_session_create(wam_c_model * model,
                         const wam_c_session_options * options,
                         wam_c_session ** session, wam_c_error * error) {
    return guarded(error, [&] {
        require(model != nullptr && model->value != nullptr,
                "model is null", "model");
        require(options != nullptr, "session options are null", "options");
        require(options->struct_size == sizeof(wam_c_session_options),
                "session options size does not match ABI", "options.struct_size");
        require(session != nullptr, "session output is null", "session");
        *session = nullptr;
        auto holder = new wam_c_session();
        try {
            holder->value = wam::session_create(
                model->value,
                {options->enable_prefix_cache != 0, options->random_seed});
        } catch (...) {
            delete holder;
            throw;
        }
        *session = holder;
    });
}

int wam_c_session_predict(wam_c_session * session,
                          const wam_c_predict_inputs * inputs,
                          wam_c_prediction * prediction,
                          wam_c_error * error) {
    return guarded(error, [&] {
        require(session != nullptr && session->value != nullptr,
                "session is null", "session");
        require(inputs != nullptr, "predict inputs are null", "inputs");
        require(prediction != nullptr, "prediction output is null", "prediction");
        std::memset(prediction, 0, sizeof(*prediction));
        wam::Inputs runtime = runtime_inputs(*inputs);
        copy_prediction(wam::predict(session->value, runtime), prediction);
    });
}

int wam_c_session_reset(wam_c_session * session, wam_c_error * error) {
    return guarded(error, [&] {
        require(session != nullptr && session->value != nullptr,
                "session is null", "session");
        const wam::Status status = wam::session_reset(session->value);
        if (!status) throw wam::Error(status.code, status.message, status.details);
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
    std::memset(prediction, 0, sizeof(*prediction));
}

void wam_c_string_free(char * value) { std::free(value); }

void wam_c_error_free(wam_c_error * error) {
    if (error == nullptr) return;
    std::free(error->message);
    std::free(error->details_json);
    clear_error(error);
}

void wam_c_session_free(wam_c_session * session) {
    if (session == nullptr) return;
    wam::session_free(session->value);
    delete session;
}

void wam_c_model_free(wam_c_model * model) {
    if (model == nullptr) return;
    wam::model_free(model->value);
    delete model;
}

} // extern "C"
