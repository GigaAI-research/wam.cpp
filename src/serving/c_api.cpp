#include "wam/c_api.h"

#include "protocol_adapter.h"
#include "wam/version.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>

struct wam_rpc_service {
    std::unique_ptr<wam::serving::ProtocolService> impl;
};

namespace {

void clear(wam_rpc_result * result) {
    if (result == nullptr) return;
    result->error_code = 0;
    result->grpc_status_code = 0;
    result->payload = nullptr;
    result->payload_size = 0;
    result->message = nullptr;
}

char * copy_string(const std::string & value) {
    if (value.empty()) return nullptr;
    char * result = static_cast<char *>(std::malloc(value.size() + 1));
    if (result == nullptr) throw std::bad_alloc();
    std::memcpy(result, value.c_str(), value.size() + 1);
    return result;
}

void copy_result(const wam::serving::RpcResult & source, wam_rpc_result * target) {
    if (target == nullptr) return;
    target->error_code = static_cast<std::uint32_t>(source.error_code);
    target->grpc_status_code = static_cast<std::uint32_t>(source.grpc_status);
    target->payload_size = source.payload.size();
    if (!source.payload.empty()) {
        target->payload = static_cast<std::uint8_t *>(std::malloc(source.payload.size()));
        if (target->payload == nullptr) throw std::bad_alloc();
        std::memcpy(target->payload, source.payload.data(), source.payload.size());
    }
    target->message = copy_string(source.message);
}

void creation_error(wam_rpc_result * result, wam::ErrorCode code,
                    wam::serving::GrpcStatusCode grpc, const std::string & message) noexcept {
    if (result == nullptr) return;
    clear(result);
    result->error_code = static_cast<std::uint32_t>(code);
    result->grpc_status_code = static_cast<std::uint32_t>(grpc);
    try { result->message = copy_string(message); } catch (...) {}
}

int create_service(const wam::ModelOptions & options, wam_rpc_service ** service,
                   wam_rpc_result * result) {
    try {
        auto created = std::make_unique<wam_rpc_service>();
        created->impl = std::make_unique<wam::serving::ProtocolService>(options);
        *service = created.release();
        return 0;
    } catch (const wam::Error & error) {
        creation_error(result, error.code(), wam::serving::grpc_status(error.code()), error.what());
    } catch (const std::exception & error) {
        creation_error(result, wam::ErrorCode::internal,
                       wam::serving::GrpcStatusCode::internal, error.what());
    } catch (...) {
        creation_error(result, wam::ErrorCode::internal,
                       wam::serving::GrpcStatusCode::internal,
                       "unknown service creation failure");
    }
    return 1;
}

} // namespace

extern "C" {

std::uint32_t wam_rpc_c_abi_version(void) {
    return wam::abi_version();
}

int wam_rpc_service_create(const wam_rpc_model_options * options,
                           wam_rpc_service ** service,
                           wam_rpc_result * result) {
    clear(result);
    if (service == nullptr || options == nullptr || options->artifact_path == nullptr) {
        creation_error(result, wam::ErrorCode::invalid_argument,
                       wam::serving::GrpcStatusCode::invalid_argument,
                       "service, options, and artifact_path are required");
        return 1;
    }
    *service = nullptr;
    try {
        wam::ModelOptions value;
        value.artifact_path = options->artifact_path;
        value.backend = static_cast<wam::Backend>(options->backend);
        value.precision = static_cast<wam::Precision>(options->precision);
        value.device_index = options->device_index;
        value.prompt_cache_capacity = options->prompt_cache_capacity;
        return create_service(value, service, result);
    } catch (...) {
        creation_error(result, wam::ErrorCode::internal,
                       wam::serving::GrpcStatusCode::internal, "invalid v1 model options");
    }
    return 1;
}

int wam_rpc_service_create_v2(const wam_rpc_model_options_v2 * options,
                              wam_rpc_service ** service,
                              wam_rpc_result * result) {
    clear(result);
    if (service == nullptr || options == nullptr ||
        options->struct_size < sizeof(wam_rpc_model_options_v2) ||
        options->artifact_path == nullptr) {
        creation_error(result, wam::ErrorCode::invalid_argument,
                       wam::serving::GrpcStatusCode::invalid_argument,
                       "valid v2 service options are required");
        return 1;
    }
    *service = nullptr;
    if ((options->fixed_token_count != 0 && options->fixed_token_ids == nullptr) ||
        (options->fixed_attention_mask_count != 0 &&
         options->fixed_attention_mask == nullptr)) {
        creation_error(result, wam::ErrorCode::invalid_argument,
                       wam::serving::GrpcStatusCode::invalid_argument,
                       "fixed prompt pointers must match their lengths");
        return 1;
    }
    try {
        wam::ModelOptions value;
        value.artifact_path = options->artifact_path;
        value.backend = static_cast<wam::Backend>(options->backend);
        value.precision = static_cast<wam::Precision>(options->precision);
        value.device_index = options->device_index;
        value.prompt_cache_capacity = options->prompt_cache_capacity;
        value.language_encoder_policy =
            static_cast<wam::LanguageEncoderPolicy>(options->language_encoder_policy);
        if (options->fixed_token_count != 0 || options->fixed_attention_mask_count != 0) {
            wam::FixedPrompt prompt;
            if (options->fixed_token_count != 0) {
                prompt.token_ids.assign(options->fixed_token_ids,
                                        options->fixed_token_ids + options->fixed_token_count);
            }
            if (options->fixed_attention_mask_count != 0) {
                prompt.attention_mask.assign(options->fixed_attention_mask,
                                             options->fixed_attention_mask +
                                                 options->fixed_attention_mask_count);
            }
            value.fixed_prompt = std::move(prompt);
        }
        return create_service(value, service, result);
    } catch (const std::exception & error) {
        creation_error(result, wam::ErrorCode::internal,
                       wam::serving::GrpcStatusCode::internal, error.what());
    } catch (...) {
        creation_error(result, wam::ErrorCode::internal,
                       wam::serving::GrpcStatusCode::internal,
                       "unknown v2 model options failure");
    }
    return 1;
}

int wam_rpc_service_call(wam_rpc_service * service,
                         std::uint32_t method,
                         const std::uint8_t * request,
                         std::size_t request_size,
                         wam_rpc_result * result) {
    if (result == nullptr) return 1;
    clear(result);
    if (service == nullptr || !service->impl) {
        creation_error(result, wam::ErrorCode::invalid_argument,
                       wam::serving::GrpcStatusCode::invalid_argument,
                       "valid RPC service is required");
        return 1;
    }
    try {
        copy_result(service->impl->call(static_cast<wam::serving::RpcMethod>(method),
                                       request, request_size), result);
        return 0;
    } catch (const std::exception & error) {
        wam_rpc_result_free(result);
        creation_error(result, wam::ErrorCode::internal,
                       wam::serving::GrpcStatusCode::internal, error.what());
    } catch (...) {
        wam_rpc_result_free(result);
        creation_error(result, wam::ErrorCode::internal,
                       wam::serving::GrpcStatusCode::internal,
                       "unknown C ABI allocation failure");
    }
    return 1;
}

void wam_rpc_result_free(wam_rpc_result * result) {
    if (result == nullptr) return;
    std::free(result->payload);
    std::free(result->message);
    clear(result);
}

void wam_rpc_service_free(wam_rpc_service * service) {
    delete service;
}

} // extern "C"
