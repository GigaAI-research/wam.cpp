#include "models/gwp05/semantics.h"

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
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        invalid("GWP structural size overflows", field, "overflow");
    }
    return left * right;
}

bool supported_normalization(policy::NormalizationKind kind) {
    return kind == policy::NormalizationKind::z_score ||
        kind == policy::NormalizationKind::min_max ||
        kind == policy::NormalizationKind::quantile;
}

} // namespace

void validate_policy_semantics(
    const policy::PolicySpecDraft & policy_spec) {
    if (policy_spec.images.views.size() != 3) {
        invalid("the current GWP-0.5 artifact contract requires three views",
                "wam.input.image.roles",
                std::to_string(policy_spec.images.views.size()));
    }
    const policy::ImageCompositionSpec & composition =
        policy_spec.images.composition;
    if (composition.kind != policy::ImageCompositionKind::canvas ||
        composition.width == 0 || composition.height == 0 ||
        composition.width % 32 != 0 || composition.height % 32 != 0) {
        invalid("GWP image composition must be a canvas divisible by 32",
                "wam.input.image.composition", "invalid canvas geometry");
    }
    if (policy_spec.images.color_space != policy::ColorSpace::rgb ||
        policy_spec.images.pixel_range !=
            policy::PixelRange::minus_one_to_one ||
        policy_spec.images.tensor_layout != policy::TensorLayout::chw) {
        invalid("GWP image output convention is unsupported",
                "wam.input.image", "expected RGB CHW in [-1,1]");
    }
    std::uint64_t covered = 0;
    for (const policy::ImagePlacement & placement : composition.placements) {
        covered += static_cast<std::uint64_t>(placement.width) *
            placement.height;
    }
    if (covered != static_cast<std::uint64_t>(composition.width) *
                       composition.height) {
        invalid("GWP image placements must cover the canvas exactly",
                "wam.input.image.composition", "canvas contains a gap");
    }
    if (policy_spec.state.model_dim != policy_spec.action.model_dim) {
        invalid("GWP state and action model dimensions must match",
                "wam.input.state.model_dim",
                "does not match wam.output.action.model_dim");
    }
    if (!supported_normalization(policy_spec.state.normalization.kind) ||
        !supported_normalization(policy_spec.action.normalization.kind)) {
        invalid("the GWP policy requires explicit normalization",
                "wam.normalization",
                "expected z_score, min_max, or quantile");
    }
    if (policy_spec.language.max_tokens == 0 ||
        policy_spec.language.max_tokens >
            static_cast<std::size_t>(kPromptTokens)) {
        invalid("GWP language capacity must be in [1,64]",
                "wam.input.language.max_tokens",
                std::to_string(policy_spec.language.max_tokens));
    }
    if (!policy_spec.language.text_encoder_in_artifact) {
        invalid("GWP artifact must contain its text encoder",
                "wam.input.language.text_encoder_in_artifact", "false");
    }
}

SequenceGeometry resolve_sequence_geometry(const StructuralConfig & config) {
    if (config.t5_capacity < kPromptTokens) {
        invalid("T5 capacity is smaller than the GWP prompt",
                "gwp05.t5_max_length", std::to_string(config.t5_capacity));
    }
    if (config.image_height <= 0 || config.image_width <= 0 ||
        config.image_height % 32 != 0 || config.image_width % 32 != 0) {
        invalid("GWP image dimensions must be positive multiples of 32",
                "image_geometry",
                std::to_string(config.image_height) + "x" +
                    std::to_string(config.image_width));
    }
    if (config.action_horizon <= 0 || config.layers <= 0 ||
        config.inference_steps <= 0) {
        invalid("GWP layer, action, and flow-step counts must be positive",
                "sequence_geometry", "non-positive count");
    }
    if (config.query_heads <= 0 ||
        config.query_heads != config.key_value_heads ||
        config.head_dim != 128 || config.hidden <= 0 ||
        config.hidden != config.query_heads * config.head_dim ||
        config.action_hidden <= 0) {
        invalid("unsupported GWP attention or expert geometry",
                "attention_geometry", "expected MHA with 128-wide heads");
    }

    SequenceGeometry geometry;
    geometry.latent_height = config.image_height / 16;
    geometry.latent_width = config.image_width / 16;
    geometry.visual_grid_height = config.image_height / 32;
    geometry.visual_grid_width = config.image_width / 32;
    geometry.visual_tokens =
        geometry.visual_grid_height * geometry.visual_grid_width;
    geometry.prefix_tokens = geometry.state_tokens + geometry.visual_tokens;
    geometry.action_tokens = config.action_horizon;
    geometry.full_tokens = geometry.prefix_tokens + geometry.action_tokens;
    return geometry;
}

TokenLayout complete_mot_layout(const SequenceGeometry & geometry) {
    if (geometry.state_tokens != 1 || geometry.visual_tokens <= 0 ||
        geometry.action_tokens <= 0 ||
        geometry.prefix_tokens !=
            geometry.state_tokens + geometry.visual_tokens ||
        geometry.full_tokens !=
            geometry.prefix_tokens + geometry.action_tokens) {
        invalid("inconsistent GWP sequence geometry", "sequence_geometry",
                "invalid counts");
    }
    TokenLayout layout;
    layout.visual_offset = 1 + geometry.action_tokens;
    layout.total_tokens = geometry.full_tokens;
    layout.prefix_tokens = geometry.prefix_tokens;
    layout.suffix_tokens = geometry.action_tokens;
    return layout;
}

std::vector<std::int32_t> action_positions(std::int64_t action_tokens,
                                           bool include_state) {
    if (action_tokens <= 0 ||
        action_tokens > std::numeric_limits<std::int32_t>::max() - 1LL) {
        invalid("action token count is out of range", "action_tokens",
                std::to_string(action_tokens));
    }
    const std::int64_t count = action_tokens + (include_state ? 1 : 0);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
        positions[static_cast<std::size_t>(index)] =
            static_cast<std::int32_t>(index + (include_state ? 0 : 1));
    }
    return positions;
}

std::vector<float> action_timesteps(std::int64_t action_tokens,
                                    float timestep,
                                    bool include_clean_state) {
    if (action_tokens <= 0 || !std::isfinite(timestep)) {
        invalid("invalid action timestep input", "timestep",
                std::to_string(timestep));
    }
    std::vector<float> result(
        static_cast<std::size_t>(action_tokens +
                                 (include_clean_state ? 1 : 0)),
        timestep);
    if (include_clean_state) result.front() = 0.0F;
    return result;
}

VisualPositions visual_positions(std::int64_t grid_height,
                                 std::int64_t grid_width) {
    if (grid_height <= 0 || grid_width <= 0 ||
        grid_height > std::numeric_limits<std::int32_t>::max() ||
        grid_width > std::numeric_limits<std::int32_t>::max()) {
        invalid("visual grid dimensions are invalid", "visual_grid",
                "outside range");
    }
    const std::size_t tokens = checked_product(
        static_cast<std::size_t>(grid_height),
        static_cast<std::size_t>(grid_width), "visual_tokens");
    VisualPositions result{
        std::vector<std::int32_t>(tokens, 0),
        std::vector<std::int32_t>(tokens),
        std::vector<std::int32_t>(tokens),
    };
    for (std::int64_t y = 0; y < grid_height; ++y) {
        for (std::int64_t x = 0; x < grid_width; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y * grid_width + x);
            result.height[index] = static_cast<std::int32_t>(y);
            result.width[index] = static_cast<std::int32_t>(x);
        }
    }
    return result;
}

std::vector<float> complete_mot_attention_mask(const TokenLayout & layout,
                                               std::int64_t heads) {
    if (heads <= 0 || layout.state_offset != 0 ||
        layout.action_offset != 1 || layout.suffix_tokens <= 0 ||
        layout.visual_offset != layout.action_offset + layout.suffix_tokens ||
        layout.total_tokens <= layout.visual_offset ||
        layout.prefix_tokens !=
            1 + layout.total_tokens - layout.visual_offset) {
        invalid("invalid complete-MoT token layout", "token_layout",
                "inconsistent offsets");
    }
    const std::size_t tokens =
        static_cast<std::size_t>(layout.total_tokens);
    const std::size_t matrix = checked_product(tokens, tokens,
                                               "attention_matrix");
    std::vector<float> result(checked_product(
        static_cast<std::size_t>(heads), matrix, "attention_mask"));
    for (std::int64_t head = 0; head < heads; ++head) {
        for (std::int64_t query = 0; query < layout.total_tokens; ++query) {
            for (std::int64_t key = 0; key < layout.total_tokens; ++key) {
                const bool query_is_action =
                    query >= layout.action_offset &&
                    query < layout.visual_offset;
                const bool key_is_action = key >= layout.action_offset &&
                    key < layout.visual_offset;
                const bool allowed = query_is_action || !key_is_action;
                const std::size_t index =
                    (static_cast<std::size_t>(head) * tokens +
                     static_cast<std::size_t>(query)) *
                        tokens +
                    static_cast<std::size_t>(key);
                result[index] = allowed
                    ? 0.0F
                    : -std::numeric_limits<float>::infinity();
            }
        }
    }
    return result;
}

PromptPlan prepare_prompt(const std::vector<std::int32_t> & token_ids,
                          const std::vector<std::int32_t> & attention_mask,
                          std::int64_t vocabulary_size,
                          policy::SequenceSide padding_side) {
    if (token_ids.empty() ||
        token_ids.size() > static_cast<std::size_t>(kPromptTokens)) {
        invalid("GWP prompt token count must be in [1,64]", "token_ids",
                std::to_string(token_ids.size()));
    }
    if (vocabulary_size <= 0) {
        invalid("T5 vocabulary size must be positive", "vocabulary_size",
                std::to_string(vocabulary_size));
    }
    if (!attention_mask.empty() &&
        attention_mask.size() != token_ids.size()) {
        invalid("prompt mask length differs from token count",
                "attention_mask", std::to_string(attention_mask.size()));
    }
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (token_ids[index] < 0 || token_ids[index] >= vocabulary_size) {
            invalid("prompt token is outside the T5 vocabulary", "token_ids",
                    std::to_string(index));
        }
    }

    std::vector<std::int32_t> mask = attention_mask;
    if (mask.empty()) mask.assign(token_ids.size(), 1);
    bool saw_valid = false;
    bool saw_padding_after_valid = false;
    for (std::size_t index = 0; index < mask.size(); ++index) {
        if (mask[index] != 0 && mask[index] != 1) {
            invalid("prompt mask values must be zero or one",
                    "attention_mask", std::to_string(index));
        }
        if (padding_side == policy::SequenceSide::right) {
            if (mask[index] == 0) saw_padding_after_valid = true;
            if (mask[index] == 1 && saw_padding_after_valid) {
                invalid("right-padded prompt mask is not contiguous",
                        "attention_mask", std::to_string(index));
            }
        } else {
            if (mask[index] == 1) saw_valid = true;
            if (mask[index] == 0 && saw_valid) {
                invalid("left-padded prompt mask is not contiguous",
                        "attention_mask", std::to_string(index));
            }
        }
    }

    PromptPlan result;
    for (std::size_t index = 0; index < mask.size(); ++index) {
        if (mask[index] == 1) result.active_token_ids.push_back(token_ids[index]);
    }
    if (result.active_token_ids.empty()) {
        invalid("prompt contains no valid tokens", "attention_mask",
                "all padding");
    }
    result.valid_tokens = result.active_token_ids.size();
    result.padded_attention_mask.assign(
        static_cast<std::size_t>(kPromptTokens), 0);
    std::fill_n(result.padded_attention_mask.begin(), result.valid_tokens, 1);
    return result;
}

std::vector<float> pad_prompt_embedding(const std::vector<float> & embedding,
                                        std::size_t input_tokens,
                                        std::size_t hidden_size) {
    if (input_tokens == 0 ||
        input_tokens > static_cast<std::size_t>(kPromptTokens) ||
        hidden_size == 0 ||
        embedding.size() != checked_product(input_tokens, hidden_size,
                                            "embedding")) {
        invalid("prompt embedding shape is invalid", "language.embedding",
                "expected [T,D] with 1 <= T <= 64");
    }
    std::vector<float> result(checked_product(
        static_cast<std::size_t>(kPromptTokens), hidden_size,
        "padded_embedding"));
    std::copy(embedding.begin(), embedding.end(), result.begin());
    return result;
}

} // namespace wam::internal::gwp05::semantics
