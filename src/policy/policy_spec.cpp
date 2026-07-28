#include "policy/policy_spec.h"

#include "models/common/gguf_reader.h"
#include "wam/error.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

namespace wam::internal::policy {
namespace {

[[noreturn]] void incompatible(const std::string & message,
                               const std::string & field,
                               const std::string & reason) {
    throw Error(ErrorCode::incompatible_artifact, message, {{field, reason}});
}

std::string image_key(const std::string & role, const char * suffix) {
    return "wam.input.image." + role + "." + suffix;
}

ResizeMode parse_resize(const std::string & value,
                        const std::string & field) {
    if (value == "none") return ResizeMode::none;
    if (value == "stretch") return ResizeMode::stretch;
    if (value == "cover_center_crop") return ResizeMode::cover_center_crop;
    incompatible("unknown image resize mode", field, value);
}

InterpolationMode parse_interpolation(const std::string & value,
                                      const std::string & field) {
    if (value == "nearest") return InterpolationMode::nearest;
    if (value == "bilinear") return InterpolationMode::bilinear;
    if (value == "bicubic") return InterpolationMode::bicubic;
    incompatible("unknown image interpolation mode", field, value);
}

ImageCompositionKind parse_composition(const std::string & value,
                                       const std::string & field) {
    if (value == "none") return ImageCompositionKind::none;
    if (value == "canvas") return ImageCompositionKind::canvas;
    incompatible("unknown image composition kind", field, value);
}

ColorSpace parse_color_space(const std::string & value,
                             const std::string & field) {
    if (value == "rgb") return ColorSpace::rgb;
    incompatible("unknown image color space", field, value);
}

PixelRange parse_pixel_range(const std::string & value,
                             const std::string & field) {
    if (value == "zero_to_one") return PixelRange::zero_to_one;
    if (value == "minus_one_to_one") return PixelRange::minus_one_to_one;
    incompatible("unknown image pixel range", field, value);
}

TensorLayout parse_layout(const std::string & value,
                          const std::string & field) {
    if (value == "chw") return TensorLayout::chw;
    if (value == "hwc") return TensorLayout::hwc;
    incompatible("unknown image tensor layout", field, value);
}

NormalizationKind parse_normalization(const std::string & value,
                                      const std::string & field) {
    if (value == "none") return NormalizationKind::none;
    if (value == "z_score") return NormalizationKind::z_score;
    if (value == "min_max") return NormalizationKind::min_max;
    if (value == "quantile") return NormalizationKind::quantile;
    incompatible("unknown normalization kind", field, value);
}

LanguageInputMode parse_language_mode(const std::string & value,
                                      const std::string & field) {
    if (value == "tokens") return LanguageInputMode::tokens;
    if (value == "embedding") return LanguageInputMode::embedding;
    if (value == "tokens_or_embedding") {
        return LanguageInputMode::tokens_or_embedding;
    }
    incompatible("unknown language input mode", field, value);
}

SequenceSide parse_sequence_side(const std::string & value,
                                 const std::string & field) {
    if (value == "left") return SequenceSide::left;
    if (value == "right") return SequenceSide::right;
    incompatible("unknown sequence side", field, value);
}

ActionRepresentation parse_action_representation(
    const std::string & value, const std::string & field) {
    if (value == "joint_position") return ActionRepresentation::joint_position;
    if (value == "eef_delta_pose") return ActionRepresentation::eef_delta_pose;
    if (value == "eef_absolute_pose") {
        return ActionRepresentation::eef_absolute_pose;
    }
    incompatible("unknown action representation", field, value);
}

ActionFrame parse_action_frame(const std::string & value,
                               const std::string & field) {
    if (value == "world") return ActionFrame::world;
    if (value == "robot_base") return ActionFrame::robot_base;
    if (value == "eef") return ActionFrame::eef;
    if (value == "controller") return ActionFrame::controller;
    incompatible("unknown action frame", field, value);
}

GripperEncoding parse_gripper(const std::string & value,
                              const std::string & field) {
    if (value == "none") return GripperEncoding::none;
    if (value == "continuous") return GripperEncoding::continuous;
    if (value == "discrete") return GripperEncoding::discrete;
    incompatible("unknown gripper encoding", field, value);
}

ActionRecoveryKind parse_recovery(const std::string & value,
                                  const std::string & field) {
    if (value == "identity") return ActionRecoveryKind::identity;
    if (value == "add_current_state") {
        return ActionRecoveryKind::add_current_state;
    }
    incompatible("unknown action recovery kind", field, value);
}

NormalizationStats read_stats(const GgufReader & reader,
                              const std::string & domain,
                              std::size_t model_dim) {
    NormalizationStats stats;
    const std::string prefix = "wam.norm." + domain + ".";
    const auto read_optional = [&](const char * name) {
        const std::string tensor_name = prefix + name;
        if (reader.find_tensor(tensor_name) == nullptr) {
            return std::vector<float>{};
        }
        reader.require_shape(tensor_name,
                             {static_cast<std::int64_t>(model_dim)});
        return reader.read_f32_tensor(tensor_name);
    };

    const bool has_current_range =
        reader.find_tensor(prefix + "lower") != nullptr ||
        reader.find_tensor(prefix + "upper") != nullptr;
    const bool has_legacy_range =
        reader.find_tensor(prefix + "q01") != nullptr ||
        reader.find_tensor(prefix + "q99") != nullptr;
    if (has_current_range && has_legacy_range) {
        incompatible("normalization range mixes current and legacy tensors",
                     prefix + "lower", "remove the q01/q99 development alias");
    }

    stats.mean = read_optional("mean");
    stats.stddev = read_optional("std");
    stats.lower = read_optional(has_current_range ? "lower" : "q01");
    stats.upper = read_optional(has_current_range ? "upper" : "q99");

    const std::vector<float> raw_mask = read_optional("mask");
    stats.mask.reserve(raw_mask.size());
    for (std::size_t index = 0; index < raw_mask.size(); ++index) {
        if (raw_mask[index] != 0.0F && raw_mask[index] != 1.0F) {
            incompatible("normalization mask must contain only zero or one",
                         prefix + "mask", "invalid value at index " +
                                                   std::to_string(index));
        }
        stats.mask.push_back(static_cast<std::uint8_t>(raw_mask[index]));
    }
    return stats;
}

bool valid_role(const std::string & role) {
    if (role.empty() || role.size() > 64 ||
        std::islower(static_cast<unsigned char>(role.front())) == 0) {
        return false;
    }
    return std::all_of(role.begin() + 1, role.end(), [](char character) {
        const unsigned char value = static_cast<unsigned char>(character);
        return std::islower(value) != 0 || std::isdigit(value) != 0 ||
               character == '_';
    });
}

void validate_fields(const std::vector<std::string> & fields,
                     std::size_t expected, const std::string & key) {
    if (fields.size() != expected) {
        incompatible("field list length does not match real dimension", key,
                     "expected " + std::to_string(expected) + ", got " +
                         std::to_string(fields.size()));
    }
    std::unordered_set<std::string> seen;
    for (const std::string & field : fields) {
        if (field.empty()) {
            incompatible("field name must not be empty", key, "empty value");
        }
        if (!seen.insert(field).second) {
            incompatible("field names must be unique", key,
                         "duplicate " + field);
        }
    }
}

bool active_dimension(const NormalizationStats & stats, std::size_t index) {
    return stats.mask.empty() || stats.mask[index] != 0;
}

void validate_stat_vector(const std::vector<float> & values,
                          std::size_t expected, const std::string & field) {
    if (values.empty()) return;
    if (values.size() != expected) {
        incompatible("normalization tensor has the wrong length", field,
                     "expected " + std::to_string(expected) + ", got " +
                         std::to_string(values.size()));
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            incompatible("normalization tensor contains a non-finite value",
                         field, "index " + std::to_string(index));
        }
    }
}

void validate_normalization(const NormalizationSpec & normalization,
                            const NormalizationStats & stats,
                            std::size_t model_dim,
                            const std::string & domain) {
    const std::string prefix = "wam.norm." + domain + ".";
    switch (normalization.kind) {
        case NormalizationKind::none:
        case NormalizationKind::z_score:
        case NormalizationKind::min_max:
        case NormalizationKind::quantile:
            break;
        default:
            incompatible("normalization kind enum is invalid",
                         "wam.normalization." + domain + ".kind",
                         "unknown enum value");
    }
    if (!std::isfinite(normalization.epsilon) ||
        normalization.epsilon < 0.0F ||
        (normalization.kind != NormalizationKind::none &&
         normalization.epsilon == 0.0F)) {
        incompatible("normalization epsilon is invalid",
                     "wam.normalization.epsilon",
                     "must be finite and positive when normalization is enabled");
    }
    if (normalization.output_clamp_lower.has_value() !=
        normalization.output_clamp_upper.has_value()) {
        incompatible("normalization output clamp must define both bounds",
                     "wam.normalization." + domain + ".output_clamp",
                     "one bound is missing");
    }
    if (normalization.output_clamp_lower.has_value() &&
        (!std::isfinite(*normalization.output_clamp_lower) ||
         !std::isfinite(*normalization.output_clamp_upper) ||
         *normalization.output_clamp_lower >=
             *normalization.output_clamp_upper)) {
        incompatible("normalization output clamp is invalid",
                     "wam.normalization." + domain + ".output_clamp",
                     "expected finite lower < upper");
    }

    validate_stat_vector(stats.mean, model_dim, prefix + "mean");
    validate_stat_vector(stats.stddev, model_dim, prefix + "std");
    validate_stat_vector(stats.lower, model_dim, prefix + "lower");
    validate_stat_vector(stats.upper, model_dim, prefix + "upper");
    if (!stats.mask.empty() && stats.mask.size() != model_dim) {
        incompatible("normalization mask has the wrong length", prefix + "mask",
                     "expected " + std::to_string(model_dim));
    }
    for (std::uint8_t value : stats.mask) {
        if (value > 1) {
            incompatible("normalization mask must be binary", prefix + "mask",
                         "value is not zero or one");
        }
    }

    if (normalization.kind == NormalizationKind::none) {
        if (normalization.clip) {
            incompatible("normalization clip requires range normalization",
                         "wam.normalization." + domain + ".clip",
                         "none normalization has no clip interval");
        }
        if (!stats.mean.empty() || !stats.stddev.empty() ||
            !stats.lower.empty() || !stats.upper.empty() || !stats.mask.empty()) {
            incompatible("normalization kind none cannot carry statistics",
                         "wam.normalization." + domain + ".kind",
                         "statistics are present");
        }
        return;
    }

    if (normalization.kind == NormalizationKind::z_score) {
        if (normalization.clip) {
            incompatible("z_score clip interval is not defined",
                         "wam.normalization." + domain + ".clip",
                         "set false; z_score has no range bounds");
        }
        if (stats.mean.empty() || stats.stddev.empty()) {
            incompatible("z_score normalization requires mean and std tensors",
                         "wam.normalization." + domain + ".kind",
                         "missing statistics");
        }
        for (std::size_t index = 0; index < model_dim; ++index) {
            if (active_dimension(stats, index) &&
                stats.stddev[index] <= normalization.epsilon) {
                incompatible("normalization std is too small", prefix + "std",
                             "index " + std::to_string(index));
            }
        }
        return;
    }

    if (stats.lower.empty() || stats.upper.empty()) {
        incompatible("range normalization requires lower and upper tensors",
                     "wam.normalization." + domain + ".kind",
                     "missing statistics");
    }
    for (std::size_t index = 0; index < model_dim; ++index) {
        if (active_dimension(stats, index) &&
            stats.upper[index] - stats.lower[index] <= normalization.epsilon) {
            incompatible("normalization range is too small", prefix + "upper",
                         "index " + std::to_string(index));
        }
    }
}

bool rectangles_overlap(const ImagePlacement & first,
                        const ImagePlacement & second) {
    const std::uint64_t first_right =
        static_cast<std::uint64_t>(first.x) + first.width;
    const std::uint64_t first_bottom =
        static_cast<std::uint64_t>(first.y) + first.height;
    const std::uint64_t second_right =
        static_cast<std::uint64_t>(second.x) + second.width;
    const std::uint64_t second_bottom =
        static_cast<std::uint64_t>(second.y) + second.height;
    return first.x < second_right && second.x < first_right &&
           first.y < second_bottom && second.y < first_bottom;
}

} // namespace

std::optional<PolicySpec> try_read_policy_spec(
    const GgufReader & reader) {
    if (!reader.has("wam.artifact_schema_version")) {
        if (reader.has("wam.policy.profile") ||
            reader.has("wam.input.image.roles") ||
            reader.has("wam.output.action.horizon")) {
            incompatible("PolicySpec schema marker is required",
                         "wam.artifact_schema_version", "missing");
        }
        return std::nullopt;
    }

    PolicySpec spec;
    spec.identity.artifact_schema_version =
        reader.require_u32("wam.artifact_schema_version");
    if (spec.identity.artifact_schema_version !=
        kPolicySpecSchemaVersion) {
        incompatible("unsupported PolicySpec schema version",
                     "wam.artifact_schema_version",
                     std::to_string(spec.identity.artifact_schema_version));
    }
    spec.identity.profile = reader.require_string("wam.policy.profile");
    spec.identity.embodiment = reader.require_string("wam.policy.embodiment");
    spec.identity.training_dataset =
        reader.require_string("wam.policy.training_dataset");
    spec.identity.checkpoint_revision =
        reader.require_string("wam.policy.checkpoint_revision");

    const std::vector<std::string> roles =
        reader.require_string_array("wam.input.image.roles");
    spec.images.views.reserve(roles.size());
    for (const std::string & role : roles) {
        ImageTransformSpec transform;
        transform.role = role;
        transform.target_height =
            reader.require_u32(image_key(role, "target_height"));
        transform.target_width =
            reader.require_u32(image_key(role, "target_width"));
        transform.resize = parse_resize(
            reader.require_string(image_key(role, "resize_mode")),
            image_key(role, "resize_mode"));
        transform.interpolation = parse_interpolation(
            reader.require_string(image_key(role, "interpolation")),
            image_key(role, "interpolation"));
        transform.antialias =
            reader.require_bool(image_key(role, "antialias"));
        spec.images.views.push_back(std::move(transform));
    }

    spec.images.composition.kind = parse_composition(
        reader.require_string("wam.input.image.composition.kind"),
        "wam.input.image.composition.kind");
    if (spec.images.composition.kind == ImageCompositionKind::canvas) {
        spec.images.composition.height =
            reader.require_u32("wam.input.image.composition.height");
        spec.images.composition.width =
            reader.require_u32("wam.input.image.composition.width");
        spec.images.composition.placements.reserve(roles.size());
        for (const std::string & role : roles) {
            const std::string key =
                "wam.input.image.composition." + role + ".rect";
            const std::vector<std::uint32_t> rect =
                reader.require_u32_array(key);
            if (rect.size() != 4) {
                incompatible("image placement must contain x, y, width, height",
                             key, "expected four values");
            }
            spec.images.composition.placements.push_back(
                {role, rect[0], rect[1], rect[2], rect[3]});
        }
    } else if (reader.has("wam.input.image.composition.height") ||
               reader.has("wam.input.image.composition.width")) {
        incompatible("composition none cannot define canvas dimensions",
                     "wam.input.image.composition.kind",
                     "canvas fields are present");
    } else {
        for (const std::string & role : roles) {
            const std::string key =
                "wam.input.image.composition." + role + ".rect";
            if (reader.has(key)) {
                incompatible("composition none cannot define placements",
                             "wam.input.image.composition.kind",
                             "canvas placement is present");
            }
        }
    }
    spec.images.color_space = parse_color_space(
        reader.require_string("wam.input.image.color_space"),
        "wam.input.image.color_space");
    spec.images.pixel_range = parse_pixel_range(
        reader.require_string("wam.input.image.pixel_range"),
        "wam.input.image.pixel_range");
    spec.images.tensor_layout = parse_layout(
        reader.require_string("wam.input.image.tensor_layout"),
        "wam.input.image.tensor_layout");

    spec.state.real_dim = reader.require_u32("wam.input.state.real_dim");
    spec.state.model_dim = reader.require_u32("wam.input.state.model_dim");
    spec.state.pad_value = reader.require_f32("wam.input.state.pad_value");
    spec.state.fields = reader.require_string_array("wam.input.state.fields");

    spec.language.input_mode = parse_language_mode(
        reader.require_string("wam.input.language.input_mode"),
        "wam.input.language.input_mode");
    spec.language.prompt_template =
        reader.require_string("wam.input.language.prompt_template");
    spec.language.tokenizer_family =
        reader.require_string("wam.input.language.tokenizer_family");
    spec.language.tokenizer_revision =
        reader.require_string("wam.input.language.tokenizer_revision");
    spec.language.max_tokens =
        reader.require_u32("wam.input.language.max_tokens");
    spec.language.text_encoder_in_artifact =
        reader.require_bool("wam.input.language.text_encoder_in_artifact");
    spec.language.padding_side = parse_sequence_side(
        reader.require_string("wam.input.language.padding_side"),
        "wam.input.language.padding_side");
    spec.language.truncation_side = parse_sequence_side(
        reader.require_string("wam.input.language.truncation_side"),
        "wam.input.language.truncation_side");
    spec.language.attention_mask_required =
        reader.require_bool("wam.input.language.attention_mask_required");
    if (reader.has("wam.input.language.special_token_ids")) {
        spec.language.special_token_ids =
            reader.require_i32_array("wam.input.language.special_token_ids");
    }

    spec.action.horizon = reader.require_u32("wam.output.action.horizon");
    spec.action.real_dim = reader.require_u32("wam.output.action.real_dim");
    spec.action.model_dim = reader.require_u32("wam.output.action.model_dim");
    spec.action.fields =
        reader.require_string_array("wam.output.action.fields");
    spec.action.representation = parse_action_representation(
        reader.require_string("wam.output.action.representation"),
        "wam.output.action.representation");
    spec.action.frame = parse_action_frame(
        reader.require_string("wam.output.action.frame"),
        "wam.output.action.frame");
    spec.action.gripper = parse_gripper(
        reader.require_string("wam.output.action.gripper"),
        "wam.output.action.gripper");
    spec.action.recovery.kind = parse_recovery(
        reader.require_string("wam.output.action.recovery.kind"),
        "wam.output.action.recovery.kind");
    if (reader.has(
            "wam.output.action.recovery.reference_state_indices")) {
        spec.action.recovery.reference_state_indices = reader.require_i32_array(
            "wam.output.action.recovery.reference_state_indices");
    } else if (spec.action.recovery.kind ==
               ActionRecoveryKind::add_current_state) {
        incompatible("add_current_state recovery requires state indices",
                     "wam.output.action.recovery.reference_state_indices",
                     "missing");
    }

    const float epsilon = reader.require_f32("wam.normalization.epsilon");
    spec.state.normalization.kind = parse_normalization(
        reader.require_string("wam.normalization.state.kind"),
        "wam.normalization.state.kind");
    spec.state.normalization.clip =
        reader.require_bool("wam.normalization.state.clip");
    if (reader.has("wam.normalization.state.output_clamp_lower")) {
        spec.state.normalization.output_clamp_lower = reader.require_f32(
            "wam.normalization.state.output_clamp_lower");
    }
    if (reader.has("wam.normalization.state.output_clamp_upper")) {
        spec.state.normalization.output_clamp_upper = reader.require_f32(
            "wam.normalization.state.output_clamp_upper");
    }
    spec.state.normalization.epsilon = epsilon;
    spec.action.normalization.kind = parse_normalization(
        reader.require_string("wam.normalization.action.kind"),
        "wam.normalization.action.kind");
    spec.action.normalization.clip =
        reader.require_bool("wam.normalization.action.clip");
    if (reader.has("wam.normalization.action.output_clamp_lower")) {
        spec.action.normalization.output_clamp_lower = reader.require_f32(
            "wam.normalization.action.output_clamp_lower");
    }
    if (reader.has("wam.normalization.action.output_clamp_upper")) {
        spec.action.normalization.output_clamp_upper = reader.require_f32(
            "wam.normalization.action.output_clamp_upper");
    }
    spec.action.normalization.epsilon = epsilon;

    spec.state.stats = read_stats(reader, "state", spec.state.model_dim);
    spec.action.stats = read_stats(reader, "action", spec.action.model_dim);

    validate_policy_spec(spec);
    return spec;
}

void validate_policy_spec(const PolicySpec & spec) {
    if (spec.identity.artifact_schema_version !=
        kPolicySpecSchemaVersion) {
        incompatible("PolicySpec schema version is invalid",
                     "wam.artifact_schema_version",
                     std::to_string(spec.identity.artifact_schema_version));
    }
    if (spec.identity.profile.empty() ||
        spec.identity.checkpoint_revision.empty() ||
        spec.identity.training_dataset.empty() ||
        spec.identity.embodiment.empty()) {
        incompatible("PolicySpec identity fields must not be empty",
                     "wam.policy", "empty identity field");
    }

    if (spec.images.views.empty()) {
        incompatible("PolicySpec requires at least one image view",
                     "wam.input.image.roles", "empty");
    }
    std::unordered_set<std::string> roles;
    for (const ImageTransformSpec & view : spec.images.views) {
        if (!valid_role(view.role)) {
            incompatible("image role has an invalid name",
                         "wam.input.image.roles", view.role);
        }
        if (!roles.insert(view.role).second) {
            incompatible("image roles must be unique",
                         "wam.input.image.roles", "duplicate " + view.role);
        }
        if (view.target_height == 0 || view.target_width == 0) {
            incompatible("image target dimensions must be positive",
                         image_key(view.role, "target_height"),
                         "zero dimension");
        }
        switch (view.resize) {
            case ResizeMode::none:
            case ResizeMode::stretch:
            case ResizeMode::cover_center_crop:
                break;
            default:
                incompatible("image resize enum is invalid",
                             image_key(view.role, "resize_mode"),
                             "unknown enum value");
        }
        switch (view.interpolation) {
            case InterpolationMode::nearest:
            case InterpolationMode::bilinear:
            case InterpolationMode::bicubic:
                break;
            default:
                incompatible("image interpolation enum is invalid",
                             image_key(view.role, "interpolation"),
                             "unknown enum value");
        }
    }

    switch (spec.images.color_space) {
        case ColorSpace::rgb:
            break;
        default:
            incompatible("image color space enum is invalid",
                         "wam.input.image.color_space", "unknown enum value");
    }
    switch (spec.images.pixel_range) {
        case PixelRange::zero_to_one:
        case PixelRange::minus_one_to_one:
            break;
        default:
            incompatible("image pixel range enum is invalid",
                         "wam.input.image.pixel_range", "unknown enum value");
    }
    switch (spec.images.tensor_layout) {
        case TensorLayout::chw:
        case TensorLayout::hwc:
            break;
        default:
            incompatible("image tensor layout enum is invalid",
                         "wam.input.image.tensor_layout", "unknown enum value");
    }
    switch (spec.images.resample_boundary) {
        case ResampleBoundaryMode::truncate:
        case ResampleBoundaryMode::clamp:
            break;
        default:
            incompatible("invalid image resample boundary mode",
                         "wam.input.image", "unknown enum value");
    }

    if (spec.images.composition.kind == ImageCompositionKind::none) {
        if (spec.images.composition.width != 0 ||
            spec.images.composition.height != 0 ||
            !spec.images.composition.placements.empty()) {
            incompatible("composition none cannot define a canvas",
                         "wam.input.image.composition.kind",
                         "canvas fields are present");
        }
    } else if (spec.images.composition.kind ==
               ImageCompositionKind::canvas) {
        if (spec.images.composition.width == 0 ||
            spec.images.composition.height == 0) {
            incompatible("canvas dimensions must be positive",
                         "wam.input.image.composition", "zero dimension");
        }
        if (spec.images.composition.placements.size() !=
            spec.images.views.size()) {
            incompatible("canvas requires exactly one placement per image role",
                         "wam.input.image.composition", "placement count mismatch");
        }
        std::unordered_set<std::string> placed;
        for (std::size_t index = 0;
             index < spec.images.composition.placements.size(); ++index) {
            const ImagePlacement & placement =
                spec.images.composition.placements[index];
            if (roles.count(placement.role) == 0 ||
                !placed.insert(placement.role).second) {
                incompatible("canvas placement role is missing or duplicated",
                             "wam.input.image.composition", placement.role);
            }
            if (placement.width == 0 || placement.height == 0 ||
                static_cast<std::uint64_t>(placement.x) + placement.width >
                    spec.images.composition.width ||
                static_cast<std::uint64_t>(placement.y) + placement.height >
                    spec.images.composition.height) {
                incompatible("canvas placement is outside the canvas",
                             "wam.input.image.composition." + placement.role +
                                 ".rect",
                             "invalid rectangle");
            }
            const ImageTransformSpec & view =
                require_image_spec(spec, placement.role);
            if (placement.width != view.target_width ||
                placement.height != view.target_height) {
                incompatible("canvas placement size differs from image transform",
                             "wam.input.image.composition." + placement.role +
                                 ".rect",
                             "size mismatch");
            }
            for (std::size_t other = 0; other < index; ++other) {
                if (rectangles_overlap(
                        placement,
                        spec.images.composition.placements[other])) {
                    incompatible("canvas placements must not overlap",
                                 "wam.input.image.composition." +
                                     placement.role + ".rect",
                                 "overlaps " +
                                     spec.images.composition.placements[other]
                                         .role);
                }
            }
        }
    } else {
        incompatible("image composition enum is invalid",
                     "wam.input.image.composition.kind",
                     "unknown enum value");
    }

    if (spec.state.real_dim == 0 ||
        spec.state.model_dim < spec.state.real_dim) {
        incompatible("state dimensions are invalid", "wam.input.state.model_dim",
                     "model_dim must be at least real_dim and both positive");
    }
    if (!std::isfinite(spec.state.pad_value)) {
        incompatible("state pad value must be finite",
                     "wam.input.state.pad_value", "non-finite");
    }
    validate_fields(spec.state.fields, spec.state.real_dim,
                    "wam.input.state.fields");
    validate_normalization(spec.state.normalization, spec.state.stats,
                           spec.state.model_dim, "state");

    switch (spec.language.input_mode) {
        case LanguageInputMode::tokens:
        case LanguageInputMode::embedding:
        case LanguageInputMode::tokens_or_embedding:
            break;
        default:
            incompatible("language input mode enum is invalid",
                         "wam.input.language.input_mode",
                         "unknown enum value");
    }
    switch (spec.language.padding_side) {
        case SequenceSide::left:
        case SequenceSide::right:
            break;
        default:
            incompatible("language padding side enum is invalid",
                         "wam.input.language.padding_side",
                         "unknown enum value");
    }
    switch (spec.language.truncation_side) {
        case SequenceSide::left:
        case SequenceSide::right:
            break;
        default:
            incompatible("language truncation side enum is invalid",
                         "wam.input.language.truncation_side",
                         "unknown enum value");
    }
    if (spec.language.max_tokens == 0) {
        incompatible("language max_tokens must be positive",
                     "wam.input.language.max_tokens", "zero");
    }
    if ((spec.language.input_mode == LanguageInputMode::tokens ||
         spec.language.input_mode == LanguageInputMode::tokens_or_embedding) &&
        !spec.language.text_encoder_in_artifact) {
        incompatible("token input requires an artifact text encoder",
                     "wam.input.language.text_encoder_in_artifact", "false");
    }
    for (std::int32_t token : spec.language.special_token_ids) {
        if (token < 0) {
            incompatible("special token ids must be non-negative",
                         "wam.input.language.special_token_ids",
                         std::to_string(token));
        }
    }

    if (spec.action.horizon == 0 || spec.action.real_dim == 0 ||
        spec.action.model_dim < spec.action.real_dim) {
        incompatible("action dimensions are invalid",
                     "wam.output.action.model_dim",
                     "horizon and dimensions must be positive; model_dim must be at least real_dim");
    }
    validate_fields(spec.action.fields, spec.action.real_dim,
                    "wam.output.action.fields");
    switch (spec.action.representation) {
        case ActionRepresentation::joint_position:
        case ActionRepresentation::eef_delta_pose:
        case ActionRepresentation::eef_absolute_pose:
            break;
        case ActionRepresentation::unknown:
        default:
            incompatible("action representation must be explicit",
                         "wam.output.action.representation",
                         "unknown enum value");
    }
    switch (spec.action.frame) {
        case ActionFrame::world:
        case ActionFrame::robot_base:
        case ActionFrame::eef:
        case ActionFrame::controller:
            break;
        case ActionFrame::unknown:
        default:
            incompatible("action frame must be explicit",
                         "wam.output.action.frame", "unknown enum value");
    }
    switch (spec.action.gripper) {
        case GripperEncoding::none:
        case GripperEncoding::continuous:
        case GripperEncoding::discrete:
            break;
        default:
            incompatible("gripper encoding enum is invalid",
                         "wam.output.action.gripper", "unknown enum value");
    }
    validate_normalization(spec.action.normalization, spec.action.stats,
                           spec.action.model_dim, "action");

    if (spec.action.recovery.kind == ActionRecoveryKind::identity) {
        if (!spec.action.recovery.reference_state_indices.empty()) {
            incompatible("identity recovery cannot define state indices",
                         "wam.output.action.recovery.reference_state_indices",
                         "unexpected values");
        }
    } else if (spec.action.recovery.kind ==
               ActionRecoveryKind::add_current_state) {
        if (spec.action.recovery.reference_state_indices.size() !=
            spec.action.real_dim) {
            incompatible("action recovery index count is invalid",
                         "wam.output.action.recovery.reference_state_indices",
                         "expected " + std::to_string(spec.action.real_dim));
        }
        for (std::int32_t index :
             spec.action.recovery.reference_state_indices) {
            if (index < -1 ||
                (index >= 0 && static_cast<std::size_t>(index) >=
                                   spec.state.real_dim)) {
                incompatible("action recovery state index is out of range",
                             "wam.output.action.recovery.reference_state_indices",
                             std::to_string(index));
            }
        }
    } else {
        incompatible("action recovery enum is invalid",
                     "wam.output.action.recovery.kind",
                     "unknown enum value");
    }
}

std::size_t policy_image_count(const PolicySpec & spec) noexcept {
    return spec.images.views.size();
}

const ImageTransformSpec & require_image_spec(const PolicySpec & spec,
                                              const std::string & role) {
    const auto found = std::find_if(
        spec.images.views.begin(), spec.images.views.end(),
        [&](const ImageTransformSpec & view) { return view.role == role; });
    if (found == spec.images.views.end()) {
        throw Error(ErrorCode::invalid_argument,
                    "image role is not declared by PolicySpec",
                    {{"image.role", role}});
    }
    return *found;
}

} // namespace wam::internal::policy
