#pragma once

#include "models/gwp05/state.h"

#include <string>
#include <vector>

namespace wam::internal::gwp05 {

struct NativeActionRegion {
    explicit NativeActionRegion(ExecutionState & state);
    ~NativeActionRegion();
    ExecutionState & state;
    bool previous = false;
};

struct ConditionOutput {
    ggml_tensor * temb = nullptr;
    ggml_tensor * modulation = nullptr;
    ggml_tensor * text = nullptr;
};

struct CrossKv {
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

struct ExpertQkv {
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

struct CachedActionBody {
    ggml_tensor * prediction = nullptr;
    ggml_tensor * action_output = nullptr;
    ggml_tensor * block0_action = nullptr;
    ggml_tensor * action_condition = nullptr;
};

std::vector<float> timestep_embedding(float timestep, std::int64_t dim);
ggml_tensor * action_encoder(ggml_context * ctx, ExecutionState & state,
                             const std::string & prefix, ggml_tensor * input);
ggml_tensor * action_decoder(ggml_context * ctx, ExecutionState & state,
                             ggml_tensor * input);
ConditionOutput build_condition(
    ggml_context * ctx, ExecutionState & state, const std::string & prefix,
    ggml_tensor * frequency, ggml_tensor * prompt, std::int64_t hidden,
    std::int64_t tokens, ggml_tensor * projected_prompt = nullptr,
    bool build_text = true);
CrossKv build_cross_kv(ggml_context * ctx, ExecutionState & state,
                       const std::string & prefix, ggml_tensor * prompt);
ggml_tensor * modulation_part(ggml_context * ctx, ggml_tensor * combined,
                              int part, std::int64_t hidden,
                              std::int64_t tokens);
ggml_tensor * modulation_add(ggml_context * ctx, ExecutionState & state,
                             ggml_tensor * modulation, ggml_tensor * table);
ggml_tensor * modulated_norm(ggml_context * ctx, ggml_tensor * input,
                             ggml_tensor * scale, ggml_tensor * shift,
                             float epsilon);
ggml_tensor * gated_residual(ggml_context * ctx, ExecutionState & state,
                             ggml_tensor * hidden, ggml_tensor * branch,
                             ggml_tensor * gate);
ExpertQkv expert_qkv(
    ggml_context * ctx, ExecutionState & state, const std::string & prefix,
    ggml_tensor * hidden, ggml_tensor * zero, ggml_tensor * scale,
    ggml_tensor * action_positions, ggml_tensor * visual_time,
    ggml_tensor * visual_height, ggml_tensor * visual_width, bool visual);
ggml_tensor * native_attention_context(
    ggml_context * ctx, ExecutionState & state, ggml_tensor * query,
    ggml_tensor * key, ggml_tensor * value, float scale);
ggml_tensor * expert_cross_ffn(
    ggml_context * ctx, ExecutionState & state, const std::string & prefix,
    ggml_tensor * hidden, ggml_tensor * prompt, ggml_tensor * scale,
    ggml_tensor * zero, ggml_tensor * gate, ggml_tensor * cached_key = nullptr,
    ggml_tensor * cached_value = nullptr);
CachedActionBody build_cached_action_body(
    ggml_context * ctx, ExecutionState & state, ggml_tensor * action_input,
    ggml_tensor * frequency_input, ggml_tensor * positions,
    ggml_tensor * dt_input, ggml_tensor * prompt_input, bool keep_debug);
bool run_mot_step(
    ExecutionState & state, const std::vector<float> & model_state,
    const std::vector<float> & action, const std::vector<float> & reference,
    const std::vector<float> & prompt, float timestep, float dt,
    bool upload_action, bool upload_prefix, std::vector<float> * prediction,
    std::vector<float> * action_output);

} // namespace wam::internal::gwp05
