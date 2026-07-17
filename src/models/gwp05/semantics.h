#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace wam::internal::gwp05::semantics {

inline constexpr std::int64_t kPromptTokens = 64;
inline constexpr std::array<const char *, 3> kCameraOrder = {
    "camera_high", "camera_left_wrist", "camera_right_wrist",
};
inline constexpr std::array<std::int32_t, 3> kVisualRopeSections = {44, 42, 42};

struct StructuralConfig {
    std::int64_t image_height = 0;
    std::int64_t image_width = 0;
    std::int64_t num_views = 0;
    std::int64_t action_chunk = 0;
    std::int64_t layers = 0;
    std::int64_t inference_steps = 0;
    std::int64_t hidden = 0;
    std::int64_t action_hidden = 0;
    std::int64_t query_heads = 0;
    std::int64_t key_value_heads = 0;
    std::int64_t head_dim = 0;
    std::int64_t t5_capacity = 0;
};

struct SequenceGeometry {
    std::int64_t latent_height = 0;
    std::int64_t latent_width = 0;
    std::int64_t visual_grid_height = 0;
    std::int64_t visual_grid_width = 0;
    std::int64_t visual_tokens = 0;
    std::int64_t prompt_tokens = kPromptTokens;
    std::int64_t state_tokens = 1;
    std::int64_t prefix_tokens = 0;
    std::int64_t action_tokens = 0;
    std::int64_t full_tokens = 0;
};

struct TokenLayout {
    std::int64_t state_offset = 0;
    std::int64_t action_offset = 1;
    std::int64_t visual_offset = 0;
    std::int64_t total_tokens = 0;
    std::int64_t prefix_tokens = 0;
    std::int64_t suffix_tokens = 0;
};

struct VisualPositions {
    std::vector<std::int32_t> time;
    std::vector<std::int32_t> height;
    std::vector<std::int32_t> width;
};

struct ImageRegion {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t width = 0;
    std::int32_t height = 0;
};

struct PromptPlan {
    std::vector<std::int32_t> active_token_ids;
    std::vector<std::int32_t> padded_attention_mask;
    std::size_t valid_tokens = 0;
};

SequenceGeometry resolve_geometry(const StructuralConfig & config);
TokenLayout complete_mot_layout(const SequenceGeometry & geometry);

std::array<std::size_t, 3> resolve_camera_order(const std::vector<std::string> & names);
std::array<ImageRegion, 3> camera_regions(std::int32_t width, std::int32_t height);
std::int32_t patch_channel_index(std::int32_t rgb_channel,
                                 std::int32_t dx, std::int32_t dy);

std::vector<float> pad_rows(const std::vector<float> & values,
                            std::size_t rows, std::size_t input_width,
                            std::size_t padded_width);
std::vector<float> normalize_and_pad_state(const std::vector<float> & state,
                                           std::size_t padded_dimension,
                                           const std::vector<float> & q01,
                                           const std::vector<float> & q99);
std::vector<float> postprocess_action(const std::vector<float> & normalized_action,
                                      std::size_t action_steps,
                                      std::size_t padded_dimension,
                                      std::size_t real_dimension,
                                      const std::vector<float> & q01,
                                      const std::vector<float> & q99,
                                      const std::vector<float> & raw_state);
bool is_delta_action_dimension(std::size_t dimension) noexcept;

std::vector<std::int32_t> action_positions(std::int64_t action_tokens,
                                           bool include_state);
std::vector<float> action_timesteps(std::int64_t action_tokens, float timestep,
                                    bool include_clean_state);
VisualPositions visual_positions(std::int64_t grid_height, std::int64_t grid_width);
std::vector<float> complete_mot_attention_mask(const TokenLayout & layout,
                                               std::int64_t heads);

PromptPlan prepare_prompt(const std::vector<std::int32_t> & token_ids,
                          const std::vector<std::int32_t> & attention_mask,
                          std::int64_t vocabulary_size);
std::vector<float> pad_prompt_embedding(const std::vector<float> & embedding,
                                        std::size_t input_tokens,
                                        std::size_t hidden_size);

} // namespace wam::internal::gwp05::semantics
