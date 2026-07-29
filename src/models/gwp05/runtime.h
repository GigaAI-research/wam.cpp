#pragma once

#include "models/gwp05/state.h"
#include "models/gwp05/semantics.h"

#include "models/common/scheduler.h"

#include <initializer_list>
#include <string>
#include <vector>

namespace wam::internal::gwp05 {

namespace gwp05_semantics = ::wam::internal::gwp05::semantics;

const char * precision_policy_name(MotPrecisionPolicy policy);
void logf(const ExecutionState & state, LogLevel level,
          const char * format, ...);
void debug_dump(const ExecutionState & state, const char * name,
                const std::vector<float> & values,
                const std::vector<std::int64_t> & shape);
bool debug_dump_enabled(const ExecutionState & state);
bool cache_debug_dump_enabled(const ExecutionState & state);
bool any_debug_dump_enabled(const ExecutionState & state);
void audit_mixed_binary_nodes(const ExecutionState & state,
                              const char * graph_name, ggml_cgraph * graph);
void debug_dump_tensor(const ExecutionState & state, const char * name,
                       ggml_tensor * tensor);

bool action_prompt_cache_enabled(const ExecutionState & state);
bool prompt_kv_cache_enabled(const ExecutionState & state);
bool single_token_timestep_enabled(const ExecutionState & state);
bool cross_request_prompt_cache_enabled(const ExecutionState & state);
bool native_bf16(const ExecutionState & state);
bool native_vae_bf16(const ExecutionState & state);
bool packed_self_qkv(const ExecutionState & state);
bool native_action_region(const ExecutionState & state);
bool tensor_shape_is(const ggml_tensor * tensor, const char * name,
                     std::initializer_list<std::int64_t> shape);
bool prefix_storage_is_valid(const ExecutionState & state);
bool init_backend(ModelResources & resources);

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * weight,
                     ggml_tensor * bias, ggml_tensor * input);
bool bf16_projection_activations(const ExecutionState & state);
bool bf16_ffn_activations(const ExecutionState & state);
bool bf16_qkv_activations(const ExecutionState & state);
bool bf16_attention_value_output(const ExecutionState & state);
bool bf16_kv_cache_storage(const ExecutionState & state);
bool unsafe_fast_native_bf16(const ExecutionState & state);
ggml_tensor * linear_with_activation(
    ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
    ggml_tensor * input, bool use_bf16);
ggml_tensor * linear_gelu_with_activation(
    ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias,
    ggml_tensor * input, bool use_bf16);
ggml_tensor * build_attention(
    ggml_context * ctx, ggml_tensor * query_input,
    ggml_tensor * key_value_input, ggml_tensor * query_weight,
    ggml_tensor * query_bias, ggml_tensor * key_weight,
    ggml_tensor * key_bias, ggml_tensor * value_weight,
    ggml_tensor * value_bias, ggml_tensor * output_weight,
    ggml_tensor * output_bias, ggml_tensor * query_norm,
    ggml_tensor * key_norm, std::int64_t heads, std::int64_t head_dim,
    ggml_tensor * mask, float scale, float eps,
    bool use_bf16_qkv = false, bool use_bf16_value_output = false);
ggml_tensor * cache_compute_tensor(ggml_context * ctx, ggml_tensor * tensor,
                                   bool use_bf16);
void set_f32_tensor(ggml_tensor * tensor, const std::vector<float> & values);
void get_f32_tensor(ggml_tensor * tensor, std::vector<float> & values);
ggml_tensor * scheduler_step(ggml_context * ctx, ExecutionState & state,
                             ggml_tensor * action, ggml_tensor * prediction,
                             ggml_tensor * dt);
ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * input,
                       ggml_tensor * weight, float eps);
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input,
                         ggml_tensor * weight, ggml_tensor * bias, float eps);

} // namespace wam::internal::gwp05
