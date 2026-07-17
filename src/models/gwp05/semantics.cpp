#include "semantics.h"

#include "wam/types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace wam::internal::gwp05::semantics {
namespace {

[[noreturn]] void invalid(const std::string & message,
                          const std::string & field,
                          const std::string & reason) {
    throw Error(ErrorCode::invalid_argument, message, {{field, reason}});
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const std::string & field) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        invalid("GWP structural size overflows", field, "overflow");
    }
    return left * right;
}

void require_finite(float value, const std::string & field, std::size_t index) {
    if (!std::isfinite(value)) {
        invalid("GWP semantic input must be finite", field, std::to_string(index));
    }
}

} // namespace

SequenceGeometry resolve_geometry(const StructuralConfig & config) {
    if (config.t5_capacity < kPromptTokens) {
        invalid("T5 capacity is smaller than the GWP prompt", "t5_capacity",
                std::to_string(config.t5_capacity));
    }
    if (config.num_views != static_cast<std::int64_t>(kCameraOrder.size())) {
        invalid("GWP-0.5 requires three camera views", "num_views",
                std::to_string(config.num_views));
    }
    if (config.image_height <= 0 || config.image_width <= 0 ||
        config.image_height % 32 != 0 || config.image_width % 32 != 0) {
        invalid("GWP image dimensions must be positive multiples of 32", "image_geometry",
                std::to_string(config.image_height) + "x" + std::to_string(config.image_width));
    }
    if (config.action_chunk <= 0 || config.layers <= 0 || config.inference_steps <= 0) {
        invalid("GWP layer, action, and flow-step counts must be positive", "sequence_geometry",
                "non-positive count");
    }
    if (config.query_heads <= 0 || config.query_heads != config.key_value_heads ||
        config.head_dim != 128 || config.hidden <= 0 ||
        config.hidden != config.query_heads * config.head_dim || config.action_hidden <= 0) {
        invalid("unsupported GWP attention or expert geometry", "attention_geometry",
                "expected MHA with 128-wide heads");
    }

    SequenceGeometry geometry;
    geometry.latent_height = config.image_height / 16;
    geometry.latent_width = config.image_width / 16;
    geometry.visual_grid_height = config.image_height / 32;
    geometry.visual_grid_width = config.image_width / 32;
    geometry.visual_tokens = geometry.visual_grid_height * geometry.visual_grid_width;
    geometry.prefix_tokens = geometry.state_tokens + geometry.visual_tokens;
    geometry.action_tokens = config.action_chunk;
    geometry.full_tokens = geometry.prefix_tokens + geometry.action_tokens;
    return geometry;
}

TokenLayout complete_mot_layout(const SequenceGeometry & geometry) {
    if (geometry.state_tokens != 1 || geometry.visual_tokens <= 0 ||
        geometry.action_tokens <= 0 ||
        geometry.prefix_tokens != geometry.state_tokens + geometry.visual_tokens ||
        geometry.full_tokens != geometry.prefix_tokens + geometry.action_tokens) {
        invalid("inconsistent GWP sequence geometry", "sequence_geometry", "invalid counts");
    }
    TokenLayout layout;
    layout.visual_offset = 1 + geometry.action_tokens;
    layout.total_tokens = geometry.full_tokens;
    layout.prefix_tokens = geometry.prefix_tokens;
    layout.suffix_tokens = geometry.action_tokens;
    return layout;
}

std::array<std::size_t, 3> resolve_camera_order(const std::vector<std::string> & names) {
    if (names.size() != kCameraOrder.size()) {
        invalid("GWP requires exactly three named camera views", "images",
                std::to_string(names.size()));
    }
    std::array<std::size_t, 3> result{};
    std::array<bool, 3> consumed{};
    for (std::size_t canonical = 0; canonical < kCameraOrder.size(); ++canonical) {
        bool found = false;
        for (std::size_t input = 0; input < names.size(); ++input) {
            if (names[input] == kCameraOrder[canonical]) {
                if (consumed[input]) invalid("camera view appears more than once", names[input], "duplicate");
                consumed[input] = true;
                result[canonical] = input;
                found = true;
                break;
            }
        }
        if (!found) invalid("required camera view is missing", kCameraOrder[canonical], "missing");
    }
    return result;
}

std::array<ImageRegion, 3> camera_regions(std::int32_t width, std::int32_t height) {
    if (width <= 0 || height <= 0) {
        invalid("camera canvas dimensions must be positive", "image_geometry",
                std::to_string(width) + "x" + std::to_string(height));
    }
    const std::int32_t top_height = height / 2;
    const std::int32_t bottom_height = height - top_height;
    const std::int32_t left_width = width / 2;
    const std::int32_t right_width = width - left_width;
    return {{
        {0, 0, width, top_height},
        {0, top_height, left_width, bottom_height},
        {left_width, top_height, right_width, bottom_height},
    }};
}

std::int32_t patch_channel_index(std::int32_t rgb_channel,
                                 std::int32_t dx, std::int32_t dy) {
    if (rgb_channel < 0 || rgb_channel >= 3 || dx < 0 || dx >= 2 || dy < 0 || dy >= 2) {
        invalid("invalid GWP 2x2 patch coordinate", "patch_coordinate", "outside range");
    }
    return (rgb_channel * 2 + dx) * 2 + dy;
}

std::vector<float> pad_rows(const std::vector<float> & values,
                            std::size_t rows, std::size_t input_width,
                            std::size_t padded_width) {
    if (rows == 0 || input_width == 0 || input_width > padded_width) {
        invalid("invalid row padding geometry", "padding", "invalid dimensions");
    }
    const std::size_t input_size = checked_product(rows, input_width, "input_size");
    if (values.size() != input_size) {
        invalid("row payload size does not match its shape", "values", std::to_string(values.size()));
    }
    std::vector<float> result(checked_product(rows, padded_width, "padded_size"), 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < input_width; ++column) {
            const float value = values[row * input_width + column];
            require_finite(value, "values", row * input_width + column);
            result[row * padded_width + column] = value;
        }
    }
    return result;
}

std::vector<float> normalize_and_pad_state(const std::vector<float> & state,
                                           std::size_t padded_dimension,
                                           const std::vector<float> & q01,
                                           const std::vector<float> & q99) {
    if (state.empty() || state.size() > padded_dimension ||
        q01.size() != padded_dimension || q99.size() != padded_dimension) {
        invalid("state/statistics dimensions are inconsistent", "state", "shape mismatch");
    }
    std::vector<float> result(padded_dimension, 0.0f);
    for (std::size_t dimension = 0; dimension < state.size(); ++dimension) {
        require_finite(state[dimension], "state", dimension);
        require_finite(q01[dimension], "state_q01", dimension);
        require_finite(q99[dimension], "state_q99", dimension);
        const float range = std::max(q99[dimension] - q01[dimension], 1.0e-8f);
        result[dimension] = ((state[dimension] - q01[dimension]) / range) * 2.0f - 1.0f;
    }
    return result;
}

bool is_delta_action_dimension(std::size_t dimension) noexcept {
    return dimension < 6 || (dimension >= 7 && dimension < 13);
}

std::vector<float> postprocess_action(const std::vector<float> & normalized_action,
                                      std::size_t action_steps,
                                      std::size_t padded_dimension,
                                      std::size_t real_dimension,
                                      const std::vector<float> & q01,
                                      const std::vector<float> & q99,
                                      const std::vector<float> & raw_state) {
    if (action_steps == 0 || real_dimension == 0 || real_dimension > padded_dimension ||
        raw_state.size() < real_dimension || q01.size() != padded_dimension ||
        q99.size() != padded_dimension ||
        normalized_action.size() != checked_product(action_steps, padded_dimension, "action_size")) {
        invalid("action/statistics dimensions are inconsistent", "action", "shape mismatch");
    }
    std::vector<float> result(checked_product(action_steps, real_dimension, "result_size"));
    for (std::size_t step = 0; step < action_steps; ++step) {
        for (std::size_t dimension = 0; dimension < real_dimension; ++dimension) {
            const std::size_t source = step * padded_dimension + dimension;
            require_finite(normalized_action[source], "normalized_action", source);
            require_finite(q01[dimension], "action_q01", dimension);
            require_finite(q99[dimension], "action_q99", dimension);
            const float range = std::max(q99[dimension] - q01[dimension], 1.0e-8f);
            float value = ((normalized_action[source] + 1.0f) * 0.5f) * range + q01[dimension];
            if (is_delta_action_dimension(dimension)) {
                require_finite(raw_state[dimension], "state", dimension);
                value += raw_state[dimension];
            }
            result[step * real_dimension + dimension] = value;
        }
    }
    return result;
}

std::vector<std::int32_t> action_positions(std::int64_t action_tokens,
                                           bool include_state) {
    if (action_tokens <= 0 || action_tokens > std::numeric_limits<std::int32_t>::max() - 1LL) {
        invalid("action token count is out of range", "action_tokens", std::to_string(action_tokens));
    }
    const std::int64_t count = action_tokens + (include_state ? 1 : 0);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
        positions[static_cast<std::size_t>(index)] =
            static_cast<std::int32_t>(index + (include_state ? 0 : 1));
    }
    return positions;
}

std::vector<float> action_timesteps(std::int64_t action_tokens, float timestep,
                                    bool include_clean_state) {
    if (action_tokens <= 0 || !std::isfinite(timestep)) {
        invalid("invalid action timestep input", "timestep", std::to_string(timestep));
    }
    std::vector<float> result(
        static_cast<std::size_t>(action_tokens + (include_clean_state ? 1 : 0)), timestep);
    if (include_clean_state) result.front() = 0.0f;
    return result;
}

VisualPositions visual_positions(std::int64_t grid_height, std::int64_t grid_width) {
    if (grid_height <= 0 || grid_width <= 0 ||
        grid_height > std::numeric_limits<std::int32_t>::max() ||
        grid_width > std::numeric_limits<std::int32_t>::max()) {
        invalid("visual grid dimensions are invalid", "visual_grid", "outside range");
    }
    const std::size_t tokens = checked_product(static_cast<std::size_t>(grid_height),
                                               static_cast<std::size_t>(grid_width),
                                               "visual_tokens");
    VisualPositions result{
        std::vector<std::int32_t>(tokens, 0),
        std::vector<std::int32_t>(tokens),
        std::vector<std::int32_t>(tokens),
    };
    for (std::int64_t y = 0; y < grid_height; ++y) {
        for (std::int64_t x = 0; x < grid_width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y * grid_width + x);
            result.height[index] = static_cast<std::int32_t>(y);
            result.width[index] = static_cast<std::int32_t>(x);
        }
    }
    return result;
}

std::vector<float> complete_mot_attention_mask(const TokenLayout & layout,
                                               std::int64_t heads) {
    if (heads <= 0 || layout.state_offset != 0 || layout.action_offset != 1 ||
        layout.suffix_tokens <= 0 ||
        layout.visual_offset != layout.action_offset + layout.suffix_tokens ||
        layout.total_tokens <= layout.visual_offset ||
        layout.prefix_tokens != 1 + layout.total_tokens - layout.visual_offset) {
        invalid("invalid complete-MoT token layout", "token_layout", "inconsistent offsets");
    }
    const std::size_t tokens = static_cast<std::size_t>(layout.total_tokens);
    const std::size_t matrix = checked_product(tokens, tokens, "attention_matrix");
    std::vector<float> result(
        checked_product(static_cast<std::size_t>(heads), matrix, "attention_mask"));
    for (std::int64_t head = 0; head < heads; ++head) {
        for (std::int64_t query = 0; query < layout.total_tokens; ++query) {
            for (std::int64_t key = 0; key < layout.total_tokens; ++key) {
                const bool query_is_action = query >= layout.action_offset &&
                    query < layout.visual_offset;
                const bool key_is_action = key >= layout.action_offset &&
                    key < layout.visual_offset;
                const bool allowed = query_is_action || !key_is_action;
                const std::size_t index =
                    (static_cast<std::size_t>(head) * tokens + query) * tokens + key;
                result[index] = allowed ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }
    }
    return result;
}

PromptPlan prepare_prompt(const std::vector<std::int32_t> & token_ids,
                          const std::vector<std::int32_t> & attention_mask,
                          std::int64_t vocabulary_size) {
    if (token_ids.empty() || token_ids.size() > static_cast<std::size_t>(kPromptTokens)) {
        invalid("GWP prompt token count must be in [1,64]", "token_ids",
                std::to_string(token_ids.size()));
    }
    if (vocabulary_size <= 0) {
        invalid("T5 vocabulary size must be positive", "vocabulary_size",
                std::to_string(vocabulary_size));
    }
    if (!attention_mask.empty() && attention_mask.size() != token_ids.size()) {
        invalid("prompt mask length differs from token count", "attention_mask",
                std::to_string(attention_mask.size()));
    }

    std::size_t valid_tokens = token_ids.size();
    if (!attention_mask.empty()) {
        valid_tokens = 0;
        bool saw_padding = false;
        for (std::size_t index = 0; index < attention_mask.size(); ++index) {
            const std::int32_t value = attention_mask[index];
            if (value != 0 && value != 1) {
                invalid("prompt mask values must be zero or one", "attention_mask",
                        std::to_string(index));
            }
            if (value == 1) {
                if (saw_padding) {
                    invalid("prompt mask must contain one contiguous valid prefix",
                            "attention_mask", std::to_string(index));
                }
                ++valid_tokens;
            } else {
                saw_padding = true;
            }
        }
    }
    if (valid_tokens == 0) invalid("prompt contains no valid tokens", "attention_mask", "all padding");

    PromptPlan result;
    result.valid_tokens = valid_tokens;
    result.active_token_ids.assign(token_ids.begin(), token_ids.begin() + valid_tokens);
    result.padded_attention_mask.assign(static_cast<std::size_t>(kPromptTokens), 0);
    std::fill_n(result.padded_attention_mask.begin(), valid_tokens, 1);
    for (std::size_t index = 0; index < valid_tokens; ++index) {
        if (result.active_token_ids[index] < 0 || result.active_token_ids[index] >= vocabulary_size) {
            invalid("prompt token is outside the T5 vocabulary", "token_ids",
                    std::to_string(index));
        }
    }
    return result;
}

std::vector<float> pad_prompt_embedding(const std::vector<float> & embedding,
                                        std::size_t input_tokens,
                                        std::size_t hidden_size) {
    if (input_tokens == 0 || input_tokens > static_cast<std::size_t>(kPromptTokens) ||
        hidden_size == 0 || embedding.size() != checked_product(input_tokens, hidden_size,
                                                                "prompt_embedding")) {
        invalid("precomputed prompt embedding has the wrong shape", "prompt_embedding",
                "expected T x hidden with T in [1,64]");
    }
    std::vector<float> result(
        checked_product(static_cast<std::size_t>(kPromptTokens), hidden_size,
                        "padded_prompt_embedding"), 0.0f);
    for (std::size_t index = 0; index < embedding.size(); ++index) {
        require_finite(embedding[index], "prompt_embedding", index);
        result[index] = embedding[index];
    }
    return result;
}

} // namespace wam::internal::gwp05::semantics
