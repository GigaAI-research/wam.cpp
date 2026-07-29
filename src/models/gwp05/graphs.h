#pragma once

#include "backends/ggml/graph_context.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <memory>
#include <vector>

namespace wam::internal::gwp05 {

using GraphContext = ggml_backend::GraphContext;

struct MotGraph : GraphContext {
    ~MotGraph() {
        if (action_input_buffer) ggml_backend_buffer_free(action_input_buffer);
    }
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * state_input = nullptr;
    ggml_tensor * action_input = nullptr;
    ggml_tensor * reference_input = nullptr;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * action_frequency = nullptr;
    ggml_tensor * visual_frequency = nullptr;
    ggml_tensor * action_positions = nullptr;
    ggml_tensor * visual_time = nullptr;
    ggml_tensor * visual_height = nullptr;
    ggml_tensor * visual_width = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * dt_input = nullptr;
    ggml_tensor * prediction = nullptr;
    ggml_tensor * action_output = nullptr;
    ggml_backend_buffer_t action_input_buffer = nullptr;
    ggml_tensor * action_tokens_debug = nullptr;
    ggml_tensor * visual_tokens_debug = nullptr;
    ggml_tensor * action_condition_debug = nullptr;
    ggml_tensor * block0_action_debug = nullptr;
};

struct VaeGraph : GraphContext {
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * pixels = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * conv_in_debug = nullptr;
    ggml_tensor * down0_debug = nullptr;
    ggml_tensor * down1_debug = nullptr;
    ggml_tensor * down2_debug = nullptr;
    ggml_tensor * down3_debug = nullptr;
};

struct PrefixGraph : GraphContext {
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * state_input = nullptr;
    ggml_tensor * reference_input = nullptr;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * action_frequency = nullptr;
    ggml_tensor * visual_frequency = nullptr;
    ggml_tensor * state_position = nullptr;
    ggml_tensor * visual_time = nullptr;
    ggml_tensor * visual_height = nullptr;
    ggml_tensor * visual_width = nullptr;
    ggml_tensor * action_prompt = nullptr;
    ggml_tensor * state_hidden_in = nullptr;
    ggml_tensor * visual_hidden_in = nullptr;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    std::vector<ggml_tensor *> prompt_keys;
    std::vector<ggml_tensor *> prompt_values;
    std::vector<ggml_tensor *> state_hidden_out;
    std::vector<ggml_tensor *> visual_hidden_out;
};

struct PrefixStorage {
    ~PrefixStorage() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * action_prompt = nullptr;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    std::vector<ggml_tensor *> prompt_keys;
    std::vector<ggml_tensor *> prompt_values;
};

struct PromptProjectionGraph : GraphContext {
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * action_prompt = nullptr;
    std::vector<ggml_tensor *> prompt_keys;
    std::vector<ggml_tensor *> prompt_values;
};

struct CachedActionGraph : GraphContext {
    ~CachedActionGraph() {
        if (action_input_buffer) ggml_backend_buffer_free(action_input_buffer);
    }
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * action_input = nullptr;
    ggml_tensor * prompt_input = nullptr;
    ggml_tensor * frequency_input = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * dt_input = nullptr;
    ggml_tensor * prediction = nullptr;
    ggml_tensor * action_output = nullptr;
    ggml_tensor * block0_action_debug = nullptr;
    ggml_tensor * block_last_action_debug = nullptr;
    ggml_tensor * action_condition_debug = nullptr;
    ggml_tensor * action_temb_debug = nullptr;
    ggml_tensor * action_q_debug = nullptr;
    ggml_tensor * action_k_debug = nullptr;
    ggml_tensor * action_v_debug = nullptr;
    ggml_backend_buffer_t action_input_buffer = nullptr;
};

struct UnrolledActionGraph : GraphContext {
    ~UnrolledActionGraph() {
        if (action_input_buffer) ggml_backend_buffer_free(action_input_buffer);
    }
    ggml_cgraph * cgraph = nullptr;
    ggml_tensor * action_input = nullptr;
    ggml_tensor * positions = nullptr;
    std::vector<ggml_tensor *> frequency_inputs;
    std::vector<ggml_tensor *> dt_inputs;
    ggml_tensor * action_output = nullptr;
    ggml_backend_buffer_t action_input_buffer = nullptr;
};

} // namespace wam::internal::gwp05
