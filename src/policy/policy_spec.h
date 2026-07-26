#pragma once

#include "wam/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wam::internal {
class GgufReader;
}

namespace wam::internal::policy {

inline constexpr std::uint32_t kPolicySpecDraftSchemaVersion = 2;

enum class ResizeMode {
    none = 0,
    stretch,
    cover_center_crop,
};

enum class InterpolationMode {
    nearest = 0,
    bilinear,
    bicubic,
};

enum class ResampleBoundaryMode {
    truncate = 0,
    clamp,
};

enum class ImageCompositionKind {
    none = 0,
    canvas,
};

enum class ColorSpace {
    rgb = 0,
};

enum class PixelRange {
    zero_to_one = 0,
    minus_one_to_one,
};

enum class TensorLayout {
    chw = 0,
    hwc,
};

enum class NormalizationKind {
    none = 0,
    z_score,
    min_max,
    quantile,
};

enum class LanguageInputMode {
    tokens = 0,
    embedding,
    tokens_or_embedding,
};

enum class SequenceSide {
    left = 0,
    right,
};

enum class ActionRepresentation {
    unknown = 0,
    joint_position,
    eef_delta_pose,
    eef_absolute_pose,
};

enum class ActionFrame {
    unknown = 0,
    world,
    robot_base,
    eef,
    controller,
};

enum class GripperEncoding {
    none = 0,
    continuous,
    discrete,
};

enum class ActionRecoveryKind {
    identity = 0,
    add_current_state,
};

struct PolicyIdentity {
    std::uint32_t artifact_schema_version = 0;
    std::string profile;
    std::string checkpoint_revision;
    std::string training_dataset;
    std::string embodiment;
};

struct ImageTransformSpec {
    std::string role;
    std::uint32_t target_height = 0;
    std::uint32_t target_width = 0;
    ResizeMode resize = ResizeMode::none;
    InterpolationMode interpolation = InterpolationMode::bilinear;
    bool antialias = false;
};

struct ImagePlacement {
    std::string role;
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct ImageCompositionSpec {
    ImageCompositionKind kind = ImageCompositionKind::none;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::vector<ImagePlacement> placements;
};

struct ImageSpec {
    std::vector<ImageTransformSpec> views;
    ImageCompositionSpec composition;
    ColorSpace color_space = ColorSpace::rgb;
    PixelRange pixel_range = PixelRange::minus_one_to_one;
    TensorLayout tensor_layout = TensorLayout::chw;
    ResampleBoundaryMode resample_boundary =
        ResampleBoundaryMode::truncate;
};

struct NormalizationSpec {
    NormalizationKind kind = NormalizationKind::none;
    bool clip = false;
    float epsilon = 0.0F;
};

struct NormalizationStats {
    std::vector<float> mean;
    std::vector<float> stddev;
    std::vector<float> q01;
    std::vector<float> q99;
    std::vector<std::uint8_t> mask;
};

struct StateSpec {
    std::vector<std::string> fields;
    std::size_t real_dim = 0;
    std::size_t model_dim = 0;
    float pad_value = 0.0F;
    NormalizationSpec normalization;
    NormalizationStats stats;
};

struct LanguageSpec {
    LanguageInputMode input_mode = LanguageInputMode::tokens;
    std::string prompt_template;
    std::string tokenizer_family;
    std::string tokenizer_revision;
    std::size_t max_tokens = 0;
    bool text_encoder_in_artifact = false;
    SequenceSide padding_side = SequenceSide::right;
    SequenceSide truncation_side = SequenceSide::right;
    bool attention_mask_required = true;
    std::vector<std::int32_t> special_token_ids;
};

struct ActionRecoverySpec {
    ActionRecoveryKind kind = ActionRecoveryKind::identity;
    std::vector<std::int32_t> reference_state_indices;
};

struct ActionSpec {
    std::size_t horizon = 0;
    std::size_t real_dim = 0;
    std::size_t model_dim = 0;
    std::vector<std::string> fields;
    ActionRepresentation representation = ActionRepresentation::unknown;
    ActionFrame frame = ActionFrame::unknown;
    GripperEncoding gripper = GripperEncoding::none;
    NormalizationSpec normalization;
    NormalizationStats stats;
    ActionRecoverySpec recovery;
};

struct PolicySpecDraft {
    PolicyIdentity identity;
    ImageSpec images;
    StateSpec state;
    LanguageSpec language;
    ActionSpec action;
};

std::optional<PolicySpecDraft> try_read_policy_spec_draft(
    const GgufReader & reader);
void validate_policy_spec_draft(const PolicySpecDraft & spec);
std::size_t policy_image_count(const PolicySpecDraft & spec) noexcept;
const ImageTransformSpec & require_image_spec(const PolicySpecDraft & spec,
                                              const std::string & role);

} // namespace wam::internal::policy
