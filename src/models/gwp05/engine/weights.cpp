#include "models/gwp05/engine/engine_internal.h"

#include <chrono>
#include <cstring>

namespace wam::internal::gwp05::engine {

const char * component_prefix(WeightComponent component) {
    switch (component) {
        case WeightComponent::mot: return "gwp.";
        case WeightComponent::t5: return "t5.";
        case WeightComponent::vae: return "vae.";
    }
    return "";
}

const char * component_name(WeightComponent component) {
    switch (component) {
        case WeightComponent::mot: return "MoT";
        case WeightComponent::t5: return "UMT5";
        case WeightComponent::vae: return "VAE";
    }
    return "unknown";
}

void refresh_component_telemetry(Gwp05ModelArch & model) {
    model.runtime_components.clear();
    model.resident_device_bytes = 0;
    for (WeightComponent component : {WeightComponent::mot, WeightComponent::t5,
                                      WeightComponent::vae}) {
        const size_t index = static_cast<size_t>(component);
        model.runtime_components.push_back({component_name(component),
                                            model.component_loaded[index],
                                            model.component_device_bytes[index],
                                            model.component_load_milliseconds[index],
                                            model.component_unload_milliseconds[index]});
        if (model.component_loaded[index]) {
            model.resident_device_bytes += model.component_device_bytes[index];
        }
    }
    model.text_encoder_resident =
        model.component_loaded[static_cast<size_t>(WeightComponent::t5)];
    model.peak_component_device_bytes = std::max(
        model.peak_component_device_bytes, model.resident_device_bytes);
}

bool component_is_loaded(const Gwp05ModelArch & model, WeightComponent component) {
    return model.component_loaded[static_cast<size_t>(component)];
}

void unload_component(Gwp05ModelArch & model, WeightComponent component) {
    const auto begin = std::chrono::steady_clock::now();
    const size_t index = static_cast<size_t>(component);
    if (!model.component_loaded[index] && !model.weight_buffers[index] &&
        !model.weight_contexts[index]) {
        return;
    }
    if (model.backend) ggml_backend_synchronize(model.backend);
    const std::string prefix = component_prefix(component);
    for (auto it = model.weights.begin(); it != model.weights.end();) {
        if (it->first.compare(0, prefix.size(), prefix) == 0) {
            it = model.weights.erase(it);
        } else {
            ++it;
        }
    }
    if (model.weight_buffers[index]) {
        ggml_backend_buffer_free(model.weight_buffers[index]);
        model.weight_buffers[index] = nullptr;
    }
    if (model.weight_contexts[index]) {
        ggml_free(model.weight_contexts[index]);
        model.weight_contexts[index] = nullptr;
    }
    model.component_loaded[index] = false;
    model.component_device_bytes[index] = 0;
    model.component_unload_milliseconds[index] =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    refresh_component_telemetry(model);
}

bool load_component(GgufReader & reader, Gwp05ModelArch & model,
                    WeightComponent component) {
    const auto begin = std::chrono::steady_clock::now();
    const size_t index = static_cast<size_t>(component);
    if (model.component_loaded[index]) return true;
    const auto & tensors = reader.tensors();
    const char * prefix = component_prefix(component);
    const size_t prefix_size = std::strlen(prefix);
    size_t weight_count = 0;
    for (const GgufTensorInfo & info : tensors) {
        if (std::strncmp(info.name.c_str(), prefix, prefix_size) == 0) ++weight_count;
    }
    if (weight_count == 0) return false;
    ggml_init_params params{};
    params.mem_size = (weight_count + 32) * ggml_tensor_overhead();
    params.no_alloc = true;
    model.weight_contexts[index] = ggml_init(params);
    if (!model.weight_contexts[index]) return false;

    size_t max_tensor_bytes = 0;
    for (const GgufTensorInfo & info : tensors) {
        const char * name = info.name.c_str();
        if (std::strncmp(name, prefix, prefix_size) != 0) continue;
        ggml_type type = GGML_TYPE_COUNT;
        if (info.dtype == DType::f32) {
            type = GGML_TYPE_F32;
        } else if (info.dtype == DType::bf16) {
            type = GGML_TYPE_BF16;
        } else {
            std::fprintf(stderr,
                         "wam(gwp05): unsupported weight dtype for %s\n", name);
            unload_component(model, component);
            return false;
        }
        ggml_tensor * destination = ggml_new_tensor(
            model.weight_contexts[index], type,
            static_cast<int>(info.shape.size()), info.shape.data());
        ggml_set_name(destination, name);
        model.weights.emplace(name, destination);
        max_tensor_bytes = std::max(max_tensor_bytes, ggml_nbytes(destination));
    }
    model.weight_buffers[index] =
        ggml_backend_alloc_ctx_tensors(model.weight_contexts[index], model.backend);
    if (!model.weight_buffers[index]) {
        std::fprintf(stderr, "wam(gwp05): cannot allocate %s weight buffer\n",
                     component_name(component));
        unload_component(model, component);
        return false;
    }

    std::vector<uint8_t> staging(max_tensor_bytes);
    size_t loaded = 0;
    for (auto & item : model.weights) {
        if (item.first.compare(0, prefix_size, prefix) != 0) continue;
        ggml_tensor * destination = item.second;
        const size_t bytes = ggml_nbytes(destination);
        if (!reader.read_tensor(item.first, staging.data(), bytes)) {
            unload_component(model, component);
            return false;
        }
        ggml_backend_tensor_set(destination, staging.data(), 0, bytes);
        if ((++loaded % 100) == 0) {
            std::fprintf(stderr, "wam(gwp05): loaded %s %zu/%zu tensors\r",
                         component_name(component), loaded, weight_count);
            std::fflush(stderr);
        }
    }
    const size_t bytes = ggml_backend_buffer_get_size(model.weight_buffers[index]);
    model.component_loaded[index] = true;
    model.component_device_bytes[index] = bytes;
    model.component_load_milliseconds[index] =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    refresh_component_telemetry(model);
    std::fprintf(stderr, "wam(gwp05): %s owns %zu tensors (%.2f GiB)\n",
                 component_name(component), weight_count,
                 static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return true;
}

bool load_resident_weights(GgufReader & reader, Gwp05ModelArch & model) {
    if (!load_component(reader, model, WeightComponent::mot) ||
        !load_component(reader, model, WeightComponent::t5) ||
        !load_component(reader, model, WeightComponent::vae)) {
        model.load_state = LoadState::failed;
        return false;
    }
    model.load_state = LoadState::fully_resident;
    std::fprintf(stderr, "wam(gwp05): loaded %zu tensors (%.2f GiB total)\n",
                 model.weights.size(), static_cast<double>(model.resident_device_bytes) /
                     (1024.0 * 1024.0 * 1024.0));
    return true;
}

} // namespace wam::internal::gwp05::engine
