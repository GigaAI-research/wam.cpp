#include "models/gwp05/runtime.h"

#include "backends/ggml/tensor_io.h"

#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

namespace wam::internal::gwp05 {

int default_cpu_threads() {
    const unsigned count = std::thread::hardware_concurrency();
    return count == 0 ? 4 : static_cast<int>(std::min(count, 8U));
}

const char * precision_policy_name(MotPrecisionPolicy policy) {
    return policy == MotPrecisionPolicy::native_bf16 ? "native-bf16" : "f32";
}

void logf(const ExecutionState & model, LogLevel level,
          const char * format, ...) {
    if (!model.logger) return;
    std::va_list arguments;
    va_start(arguments, format);
    model.logger->logv(level, format, arguments);
    va_end(arguments);
}

void debug_dump(const ExecutionState & model, const char * name,
                const std::vector<float> & values,
                const std::vector<int64_t> & shape) {
    if (model.debug_dumper) model.debug_dumper->write(name, values, shape);
}

bool debug_dump_enabled(const ExecutionState & model) {
    return model.debug_dumper && model.debug_dumper->enabled();
}

bool cache_debug_dump_enabled(const ExecutionState & model) {
    return debug_dump_enabled(model);
}

bool any_debug_dump_enabled(const ExecutionState & model) {
    return debug_dump_enabled(model);
}

void audit_mixed_binary_nodes(const ExecutionState & model,
                              const char * graph_name, ggml_cgraph * graph) {
    if (model.debug_dumper) {
        model.debug_dumper->audit_mixed_binary_nodes(graph_name, graph);
    }
}

void debug_dump_tensor(const ExecutionState & model, const char * name,
                       ggml_tensor * tensor) {
    if (model.debug_dumper) model.debug_dumper->write_tensor(name, tensor);
}

void replace_all(std::string & value, const char * source, const char * target) {
    const size_t source_size = std::strlen(source);
    const size_t target_size = std::strlen(target);
    size_t position = 0;
    while ((position = value.find(source, position)) != std::string::npos) {
        value.replace(position, source_size, target);
        position += target_size;
    }
}

std::string compact_tensor_name(const std::string & full_name) {
    const size_t dot = full_name.find('.');
    if (dot == std::string::npos) return full_name;
    const std::string component = full_name.substr(0, dot);
    std::string name = full_name.substr(dot + 1);
    auto apply = [&](std::initializer_list<std::pair<const char *, const char *>> replacements) {
        for (const auto & item : replacements) replace_all(name, item.first, item.second);
    };
    if (component == "gwp") {
        apply({
            {"action_condition_embedder.", "acond."}, {"condition_embedder.", "cond."},
            {"patch_embedding.", "patch."}, {"state_encoder.", "state."},
            {"action_encoder.", "action."}, {"action_decoder.", "decode."},
            {"blocks.", "blk."}, {".action_expert.", ".a."},
            {".visual_expert.", ".v."}, {"time_embedder.linear_1.", "time.0."},
            {"time_embedder.linear_2.", "time.1."}, {"text_embedder.linear_1.", "text.0."},
            {"text_embedder.linear_2.", "text.1."}, {"time_proj.", "time.out."},
            {".ffn.net.0.proj.", ".ffn.in."}, {".ffn.net.2.", ".ffn.out."},
            {".attn1.", ".sa."}, {".attn2.", ".ca."},
            {".to_out.0.", ".o."}, {".to_qkv.", ".qkv."}, {".to_q.", ".q."},
            {".to_k.", ".k."}, {".to_v.", ".vproj."},
            {".norm_q.", ".qn."}, {".norm_k.", ".kn."},
            {".norm2.", ".ca_norm."}, {"in_proj.", "in."},
            {"mid_proj.", "mid."}, {"out_proj.", "out."},
            {"scale_shift_table", "mod"},
        });
    } else if (component == "t5") {
        apply({
            {"shared.", "token_embd."}, {"encoder.final_layer_norm.", "final_norm."},
            {"encoder.block.", "blk."}, {".layer.0.SelfAttention.", ".sa."},
            {".layer.0.layer_norm.", ".sa_norm."},
            {".layer.1.DenseReluDense.", ".ffn."},
            {".layer.1.layer_norm.", ".ffn_norm."},
            {"relative_attention_bias.", "rel."}, {"wi_0.", "wi0."},
            {"wi_1.", "wi1."},
        });
    } else if (component == "vae") {
        apply({
            {"encoder.", "enc."}, {"down_blocks.", "down."},
            {"downsampler.resample.1.", "ds."}, {"resnets.", "res."},
            {"mid_block.", "mid."}, {"attentions.0.", "attn."},
            {"conv_shortcut.", "skip."}, {"conv_in.", "in."},
            {"conv_out.", "out."}, {"norm_out.gamma", "out_norm"},
            {"to_qkv.", "qkv."}, {"quant_conv.", "quant."},
            {"norm1.gamma", "n1"}, {"norm2.gamma", "n2"},
            {"norm.gamma", "norm"}, {"conv1.", "c1."}, {"conv2.", "c2."},
        });
    } else {
        return full_name;
    }
    return component + "." + name;
}

bool action_prompt_cache_enabled(const ExecutionState & model) {
    return model.tuning.action_prompt_cache;
}

bool prompt_kv_cache_enabled(const ExecutionState & model) {
    return action_prompt_cache_enabled(model) && model.tuning.prompt_kv_cache;
}

bool single_token_timestep_enabled(const ExecutionState & model) {
    return model.dispatch.single_token_timestep;
}

bool cross_request_prompt_cache_enabled(const ExecutionState & model) {
    return prompt_kv_cache_enabled(model) && model.prompt_cache_limit != 0;
}

bool native_bf16(const ExecutionState & model) {
    return model.dispatch.native_bf16;
}

bool native_vae_bf16(const ExecutionState & model) {
    return model.dispatch.bf16_vae;
}

bool packed_self_qkv(const ExecutionState & model) {
    return model.dispatch.packed_qkv;
}

bool native_action_region(const ExecutionState & model) {
    return native_bf16(model) && model.building_native_action;
}


bool tensor_shape_is(const ggml_tensor * tensor, const char * name,
                     std::initializer_list<int64_t> shape) {
    (void) name;
    if (!tensor) {
        return false;
    }
    size_t dimension = 0;
    for (int64_t expected : shape) {
        if (tensor->ne[dimension] != expected) {
            return false;
        }
        ++dimension;
    }
    for (; dimension < GGML_MAX_DIMS; ++dimension) {
        if (tensor->ne[dimension] != 1) {
            return false;
        }
    }
    return true;
}

bool bf16_attention_value_output(const ExecutionState & model);
bool bf16_kv_cache_storage(const ExecutionState & model);

bool prefix_storage_is_valid(const ExecutionState & model) {
    const ModelGeometry & cfg = model.cfg;
    const PrefixStorage * storage = model.prefix_storage.get();
    if (!storage) {
        logf(model, LogLevel::error, "gwp05: cached action requested without prefix storage");
        return false;
    }
    const size_t layers = static_cast<size_t>(cfg.n_layers);
    const ggml_type expected_key_type = bf16_kv_cache_storage(model)
        ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    const ggml_type expected_value_type = (bf16_kv_cache_storage(model) ||
                                           bf16_attention_value_output(model))
        ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    if (storage->keys.size() != layers || storage->values.size() != layers) {
        logf(model, LogLevel::error, "gwp05: prefix K/V layer count does not match model layers");
        return false;
    }
    if (action_prompt_cache_enabled(model) &&
        !tensor_shape_is(storage->action_prompt, "projected action prompt",
                         {cfg.expert_h, cfg.n_lang})) {
        return false;
    }
    if (action_prompt_cache_enabled(model) && storage->action_prompt->type !=
            (native_bf16(model) ? GGML_TYPE_BF16 : GGML_TYPE_F32)) {
        logf(model, LogLevel::error, "gwp05: projected action prompt dtype mismatches policy");
        return false;
    }
    if (prompt_kv_cache_enabled(model) &&
        (storage->prompt_keys.size() != layers || storage->prompt_values.size() != layers)) {
        logf(model, LogLevel::error, "gwp05: prompt K/V layer count does not match model layers");
        return false;
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        if (!tensor_shape_is(storage->keys[layer], "prefix key",
                             {cfg.head_dim, cfg.n_prefix, cfg.n_q_heads}) ||
            !tensor_shape_is(storage->values[layer], "prefix value",
                             {cfg.n_prefix, cfg.head_dim, cfg.n_q_heads}) ||
            storage->keys[layer]->type != expected_key_type ||
            storage->values[layer]->type != expected_value_type) {
            return false;
        }
        if (prompt_kv_cache_enabled(model) &&
            (!tensor_shape_is(storage->prompt_keys[layer], "prompt key",
                              {cfg.head_dim, cfg.n_lang, cfg.n_q_heads}) ||
             !tensor_shape_is(storage->prompt_values[layer], "prompt value",
                              {cfg.n_lang, cfg.head_dim, cfg.n_q_heads}) ||
             storage->prompt_keys[layer]->type != expected_key_type ||
             storage->prompt_values[layer]->type != expected_value_type)) {
            return false;
        }
    }
    return true;
}

bool init_backend(ModelResources & model) {
#ifdef GGML_USE_CUDA
    if (model.backend_request == Backend::automatic ||
        model.backend_request == Backend::cuda) {
        model.backend_context.reset(
            ggml_backend_cuda_init(model.device_index));
        model.backend = model.backend_context.get();
    }
    if (model.backend) logf(model, LogLevel::info, "gwp05: using CUDA backend");
#endif
#ifdef GGML_USE_METAL
    if (!model.backend) {
        model.backend_context.reset(ggml_backend_metal_init());
        model.backend = model.backend_context.get();
    }
    if (model.backend) logf(model, LogLevel::info, "gwp05: using Metal backend");
#endif
    if (!model.backend && model.backend_request == Backend::cuda) {
        logf(model, LogLevel::error,
                     "gwp05: requested CUDA backend is unavailable");
        return false;
    }
    if (!model.backend && model.backend_request == Backend::automatic) {
        model.backend_context.reset(ggml_backend_cpu_init());
        model.backend = model.backend_context.get();
        if (!model.backend) return false;
        ggml_backend_cpu_set_n_threads(model.backend, model.n_threads);
        logf(model, LogLevel::info, "gwp05: using CPU backend (%d threads; debug only)", model.n_threads);
    }
    const char * backend_name = ggml_backend_name(model.backend);
    const bool cuda = backend_name && std::strstr(backend_name, "CUDA") != nullptr;
    if (native_bf16(model) && !cuda) {
        logf(model, LogLevel::error,
                     "gwp05: native-bf16 requires the validated CUDA backend; "
                     "no implicit mixed or CPU fallback is allowed");
        return false;
    }
#ifndef WAM_GWP05_CUDNN
    if (native_vae_bf16(model)) {
        logf(model, LogLevel::error,
                     "gwp05: BF16 VAE policy requires a build configured "
                     "with WAM_GWP05_CUDNN=ON");
        return false;
    }
#endif
    logf(model, LogLevel::error,
        "gwp05: execution precision=%s weights=%s hidden=%s cache=%s "
        "vae=%s math=%s attention=%s denoise=%s "
        "graph_identity=stable fallback=none",
        precision_policy_name(model.precision_policy),
        native_bf16(model) ? "BF16" : "F32",
        native_bf16(model) ? "BF16" : "F32",
        native_bf16(model) ? "BF16" : "F32",
        native_vae_bf16(model) ? "BF16-convolution/F32-norm" : "F32",
        "f32-sensitive-islands", "explicit",
        model.dispatch.unrolled_denoise ? "single-token-unrolled" : "one-step-replay");
    return true;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
                     ggml_tensor * input) {
    ggml_tensor * result = ggml_mul_mat(ctx, weight, input);
    if (bias && bias->type != result->type) {
        bias = ggml_cast(ctx, bias, result->type);
    }
    return bias ? ggml_add(ctx, result, bias) : result;
}

bool bf16_projection_activations(const ExecutionState & model) {
    return native_action_region(model) && model.dispatch.bf16_output_gemm;
}

bool bf16_ffn_activations(const ExecutionState & model) {
    return native_action_region(model) && model.dispatch.bf16_output_gemm;
}

bool bf16_qkv_activations(const ExecutionState & model) {
    return native_action_region(model) && model.dispatch.bf16_output_gemm;
}

bool bf16_attention_value_output(const ExecutionState & model) {
    return native_action_region(model);
}

bool bf16_kv_cache_storage(const ExecutionState & model) {
    return model.dispatch.bf16_hidden_and_kv;
}

bool unsafe_fast_native_bf16(const ExecutionState & model) {
    return native_action_region(model) && !model.dispatch.f32_accumulation;
}

ggml_tensor * linear_with_activation(ggml_context * ctx, ggml_tensor * weight,
                                     ggml_tensor * bias, ggml_tensor * input,
                                     bool use_bf16) {
    if (use_bf16 && input->type != GGML_TYPE_BF16) {
        input = ggml_cast(ctx, input, GGML_TYPE_BF16);
    }
    if (use_bf16 && weight->type == GGML_TYPE_BF16) {
        ggml_tensor * result = ggml_mul_mat_out_type(
            ctx, weight, input, GGML_TYPE_BF16);
        return bias ? ggml_add(ctx, result, bias) : result;
    }
    return linear(ctx, weight, bias, input);
}

ggml_tensor * linear_gelu_with_activation(ggml_context * ctx,
                                          ggml_tensor * weight,
                                          ggml_tensor * bias,
                                          ggml_tensor * input,
                                          bool use_bf16) {
    if (use_bf16 && weight->type == GGML_TYPE_BF16 && bias &&
        bias->type == GGML_TYPE_BF16) {
        if (input->type != GGML_TYPE_BF16) {
            input = ggml_cast(ctx, input, GGML_TYPE_BF16);
        }
        ggml_tensor * projected = ggml_mul_mat_out_type(
            ctx, weight, input, GGML_TYPE_BF16);
        return ggml_bias_gelu_bf16(ctx, projected, bias);
    }
    return ggml_gelu(ctx, linear_with_activation(
        ctx, weight, bias, input, use_bf16));
}

ggml_tensor * cache_compute_tensor(ggml_context * ctx, ggml_tensor * tensor,
                                   bool use_bf16) {
    if (!tensor) return nullptr;
    const ggml_type target = use_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    return tensor->type == target ? tensor : ggml_cast(ctx, tensor, target);
}

void set_f32_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    ggml_backend::set_f32(tensor, values);
}

void get_f32_tensor(ggml_tensor * tensor, std::vector<float> & values) {
    values = ggml_backend::get_f32(tensor);
}

ggml_tensor * scheduler_step(ggml_context * ctx, ExecutionState & model,
                             ggml_tensor * action, ggml_tensor * velocity,
                             ggml_tensor * dt) {
    if (!native_action_region(model)) {
        return ggml_add(ctx, action, ggml_mul(ctx, velocity, dt));
    }
    if (unsafe_fast_native_bf16(model)) {
        if (action->type != GGML_TYPE_BF16) action = ggml_cast(ctx, action, GGML_TYPE_BF16);
        if (velocity->type != GGML_TYPE_BF16) velocity = ggml_cast(ctx, velocity, GGML_TYPE_BF16);
        if (dt->type != GGML_TYPE_BF16) dt = ggml_cast(ctx, dt, GGML_TYPE_BF16);
        return ggml_add(ctx, action, ggml_mul(ctx, velocity, dt));
    }
    ggml_tensor * action_f32 = ggml_cast(ctx, action, GGML_TYPE_F32);
    ggml_tensor * velocity_f32 = ggml_cast(ctx, velocity, GGML_TYPE_F32);
    return ggml_cast(
        ctx, ggml_add(ctx, action_f32, ggml_mul(ctx, velocity_f32, dt)),
        GGML_TYPE_BF16);
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * input,
                       ggml_tensor * weight, float eps) {
    if (input->type == GGML_TYPE_BF16 && weight && weight->type == GGML_TYPE_BF16) {
        return ggml_rms_norm_bf16_f32(ctx, input, weight, eps);
    }
    if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx, input, GGML_TYPE_F32);
    ggml_tensor * result = ggml_rms_norm(ctx, input, eps);
    if (!weight) return result;
    ggml_tensor * scale = ggml_cast(ctx, weight, GGML_TYPE_F32);
    if (ggml_n_dims(input) == 3 && ggml_n_dims(scale) == 1) {
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, 1);
    }
    return ggml_mul(ctx, result, scale);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input,
                         ggml_tensor * weight, ggml_tensor * bias, float eps) {
    if (input->type == GGML_TYPE_BF16 && weight && bias &&
        weight->type == GGML_TYPE_BF16 && bias->type == GGML_TYPE_BF16) {
        return ggml_layer_norm_bf16(ctx, input, weight, bias, eps);
    }
    if (input->type != GGML_TYPE_F32) input = ggml_cast(ctx, input, GGML_TYPE_F32);
    ggml_tensor * result = ggml_norm(ctx, input, eps);
    if (weight) result = ggml_mul(ctx, result, ggml_cast(ctx, weight, GGML_TYPE_F32));
    if (bias) result = ggml_add(ctx, result, ggml_cast(ctx, bias, GGML_TYPE_F32));
    return result;
}

} // namespace wam::internal::gwp05
