#include "models/fastwam/semantics.h"

#include "models/fastwam/artifact.h"
#include "wam/error.h"

#include <limits>
#include <string>

namespace wam::internal::fastwam::semantics {
namespace {

[[noreturn]] void incompatible(const std::string & message,
                               const std::string & field,
                               const std::string & reason) {
    throw Error(ErrorCode::incompatible_artifact, message, {{field, reason}});
}

} // namespace

SequenceGeometry resolve_geometry(const GeometryInput & input) {
    if (input.spatial_downsample == 0 || input.image_height == 0 ||
        input.image_width == 0 || input.action_horizon == 0 ||
        input.num_layers == 0 || input.num_heads == 0 || input.head_dim == 0) {
        incompatible("FastWAM sequence geometry contains zero dimensions",
                     "fastwam.geometry", "all dimensions must be positive");
    }
    constexpr std::uint32_t video_patch = 2;
    const std::uint32_t total_stride =
        input.spatial_downsample * video_patch;
    if (input.image_height % total_stride != 0 ||
        input.image_width % total_stride != 0) {
        incompatible("FastWAM image geometry cannot form video tokens",
                     "fastwam.image_height",
                     "dimensions must be divisible by VAE scale times patch size");
    }
    if (input.video_hidden_dim != input.num_heads * input.head_dim) {
        incompatible("FastWAM attention geometry is inconsistent",
                     "fastwam.video_hidden_dim", "hidden != heads * head_dim");
    }

    SequenceGeometry result;
    result.latent_height = input.image_height / input.spatial_downsample;
    result.latent_width = input.image_width / input.spatial_downsample;
    result.video_grid_height = result.latent_height / video_patch;
    result.video_grid_width = result.latent_width / video_patch;
    const std::uint64_t tokens =
        static_cast<std::uint64_t>(result.video_grid_height) *
        result.video_grid_width;
    if (tokens == 0 || tokens > std::numeric_limits<std::uint32_t>::max()) {
        incompatible("FastWAM visual token count is invalid",
                     "fastwam.video_tokens", std::to_string(tokens));
    }
    result.video_tokens = static_cast<std::uint32_t>(tokens);
    result.action_tokens = input.action_horizon;
    return result;
}

VisualPositions visual_positions(const SequenceGeometry & geometry) {
    VisualPositions result;
    result.time.reserve(geometry.video_tokens);
    result.height.reserve(geometry.video_tokens);
    result.width.reserve(geometry.video_tokens);
    for (std::uint32_t height = 0; height < geometry.video_grid_height;
         ++height) {
        for (std::uint32_t width = 0; width < geometry.video_grid_width;
             ++width) {
            result.time.push_back(0);
            result.height.push_back(static_cast<std::int32_t>(height));
            result.width.push_back(static_cast<std::int32_t>(width));
        }
    }
    return result;
}

std::vector<std::int32_t> action_positions(
    const SequenceGeometry & geometry) {
    std::vector<std::int32_t> result(geometry.action_tokens);
    for (std::uint32_t index = 0; index < geometry.action_tokens; ++index) {
        result[index] = static_cast<std::int32_t>(index);
    }
    return result;
}

void validate_policy_semantics(const policy::PolicySpec & spec,
                               const ArtifactContract & artifact) {
    const Geometry & geometry = artifact.geometry;
    if (spec.images.views.size() != geometry.num_cameras ||
        spec.images.composition.kind != policy::ImageCompositionKind::canvas ||
        spec.images.composition.height != geometry.image_height ||
        spec.images.composition.width != geometry.image_width) {
        incompatible("FastWAM image PolicySpec differs from checkpoint geometry",
                     "wam.input.image", "view count or canvas geometry mismatch");
    }
    if (spec.state.real_dim != geometry.proprio_dim ||
        spec.state.model_dim != geometry.proprio_dim) {
        incompatible("FastWAM state PolicySpec differs from checkpoint geometry",
                     "wam.input.state.model_dim", "proprio dimension mismatch");
    }
    if (spec.action.horizon != geometry.action_horizon ||
        spec.action.real_dim != geometry.action_dim ||
        spec.action.model_dim != geometry.action_dim) {
        incompatible("FastWAM action PolicySpec differs from checkpoint geometry",
                     "wam.output.action", "action dimension or horizon mismatch");
    }
    if (spec.language.input_mode != policy::LanguageInputMode::embedding ||
        spec.language.text_encoder_in_artifact ||
        spec.language.max_tokens != geometry.context_len) {
        incompatible("FastWAM Gate B requires external embedding language input",
                     "wam.input.language", "embedding-only contract mismatch");
    }
    const auto state_normalization = spec.state.normalization.kind;
    const auto action_normalization = spec.action.normalization.kind;
    if (state_normalization != action_normalization ||
        (state_normalization != policy::NormalizationKind::min_max &&
         state_normalization != policy::NormalizationKind::z_score)) {
        incompatible("FastWAM requires matching min-max or z-score normalization",
                     "wam.normalization", "profile/checkpoint mismatch");
    }
    if (spec.action.recovery.kind != policy::ActionRecoveryKind::identity) {
        incompatible("selected FastWAM Gate B profile requires identity action recovery",
                     "wam.output.action.recovery.kind", "profile mismatch");
    }
}

} // namespace wam::internal::fastwam::semantics
