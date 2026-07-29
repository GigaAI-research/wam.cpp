#include "bindings/metadata_codec.h"

#include "wam/version.h"

#include <iomanip>
#include <sstream>
#include <type_traits>

namespace wam::bindings {
namespace {

std::string quote(const std::string & value) {
    std::ostringstream output;
    output << '"';
    for (unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<unsigned>(character)
                           << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    output << '"';
    return output.str();
}

template <typename T>
void numeric_array(std::ostringstream & output, const std::vector<T> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        if constexpr (std::is_same_v<T, std::uint8_t>) {
            output << static_cast<unsigned>(values[index]);
        } else {
            output << values[index];
        }
    }
    output << ']';
}

void string_array(std::ostringstream & output,
                  const std::vector<std::string> & values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << quote(values[index]);
    }
    output << ']';
}

const char * backend_name(Backend value) {
    switch (value) {
        case Backend::automatic: return "automatic";
        case Backend::cpu: return "cpu";
        case Backend::cuda: return "cuda";
        case Backend::cpu_metadata: return "cpu_metadata";
        case Backend::unknown: return "unknown";
    }
    return "unknown";
}

const char * precision_name(ComputePrecision value) {
    switch (value) {
        case ComputePrecision::automatic: return "automatic";
        case ComputePrecision::f32: return "f32";
        case ComputePrecision::f16: return "f16";
        case ComputePrecision::bf16: return "bf16";
        case ComputePrecision::fp8_e4m3: return "fp8_e4m3";
        case ComputePrecision::fp8_e5m2: return "fp8_e5m2";
        case ComputePrecision::int8: return "int8";
        case ComputePrecision::unknown: return "unknown";
    }
    return "unknown";
}

const char * language_runtime_name(LanguageRuntimeMode value) {
    switch (value) {
        case LanguageRuntimeMode::automatic: return "automatic";
        case LanguageRuntimeMode::tokens: return "tokens";
        case LanguageRuntimeMode::external_embedding: return "external_embedding";
    }
    return "unknown";
}

template <typename E>
const char * enum_name(E value);

#define WAM_ENUM_NAMES(Type, ...)                                      \
    template <> const char * enum_name(Type value) {                   \
        static const char * names[] = {__VA_ARGS__};                   \
        const auto index = static_cast<std::size_t>(value);            \
        return index < sizeof(names) / sizeof(names[0])                \
            ? names[index] : "unknown";                               \
    }

WAM_ENUM_NAMES(ResizeMode, "none", "stretch", "cover_center_crop")
WAM_ENUM_NAMES(InterpolationMode, "nearest", "bilinear", "bicubic")
WAM_ENUM_NAMES(ResampleBoundaryMode, "truncate", "clamp")
WAM_ENUM_NAMES(ImageCompositionKind, "none", "canvas")
WAM_ENUM_NAMES(ColorSpace, "rgb")
WAM_ENUM_NAMES(PixelRange, "zero_to_one", "minus_one_to_one")
WAM_ENUM_NAMES(TensorLayout, "chw", "hwc")
WAM_ENUM_NAMES(NormalizationKind, "none", "z_score", "min_max", "quantile")
WAM_ENUM_NAMES(LanguageInputMode, "tokens", "embedding", "tokens_or_embedding")
WAM_ENUM_NAMES(SequenceSide, "left", "right")
WAM_ENUM_NAMES(ActionRepresentation, "unknown", "joint_position", "eef_delta_pose", "eef_absolute_pose")
WAM_ENUM_NAMES(ActionFrame, "unknown", "world", "robot_base", "eef", "controller")
WAM_ENUM_NAMES(GripperEncoding, "none", "continuous", "discrete")
WAM_ENUM_NAMES(ActionRecoveryKind, "identity", "add_current_state")
#undef WAM_ENUM_NAMES

void normalization_json(std::ostringstream & output,
                        const NormalizationSpec & spec,
                        const NormalizationStats & stats) {
    output << "{\"kind\":" << quote(enum_name(spec.kind))
           << ",\"clip\":" << (spec.clip ? "true" : "false")
           << ",\"epsilon\":" << spec.epsilon
           << ",\"mean\":";
    numeric_array(output, stats.mean);
    output << ",\"stddev\":";
    numeric_array(output, stats.stddev);
    output << ",\"lower\":";
    numeric_array(output, stats.lower);
    output << ",\"upper\":";
    numeric_array(output, stats.upper);
    output << ",\"mask\":";
    numeric_array(output, stats.mask);
    if (spec.output_clamp_lower.has_value()) {
        output << ",\"output_clamp_lower\":"
               << *spec.output_clamp_lower
               << ",\"output_clamp_upper\":"
               << *spec.output_clamp_upper;
    }
    output << '}';
}

void policy_json(std::ostringstream & output, const PolicySpec & spec) {
    output << "{\"artifact_schema_version\":"
           << spec.identity.artifact_schema_version
           << ",\"profile\":" << quote(spec.identity.profile)
           << ",\"checkpoint_revision\":" << quote(spec.identity.checkpoint_revision)
           << ",\"training_dataset\":" << quote(spec.identity.training_dataset)
           << ",\"embodiment\":" << quote(spec.identity.embodiment)
           << ",\"images\":{\"views\":[";
    for (std::size_t index = 0; index < spec.images.views.size(); ++index) {
        const auto & view = spec.images.views[index];
        if (index != 0) output << ',';
        output << "{\"role\":" << quote(view.role)
               << ",\"target_height\":" << view.target_height
               << ",\"target_width\":" << view.target_width
               << ",\"resize\":" << quote(enum_name(view.resize))
               << ",\"interpolation\":" << quote(enum_name(view.interpolation))
               << ",\"antialias\":" << (view.antialias ? "true" : "false")
               << '}';
    }
    output << "],\"composition\":{\"kind\":"
           << quote(enum_name(spec.images.composition.kind))
           << ",\"height\":" << spec.images.composition.height
           << ",\"width\":" << spec.images.composition.width
           << ",\"placements\":[";
    for (std::size_t index = 0;
         index < spec.images.composition.placements.size(); ++index) {
        const auto & item = spec.images.composition.placements[index];
        if (index != 0) output << ',';
        output << "{\"role\":" << quote(item.role)
               << ",\"x\":" << item.x << ",\"y\":" << item.y
               << ",\"width\":" << item.width
               << ",\"height\":" << item.height << '}';
    }
    output << "]},\"color_space\":" << quote(enum_name(spec.images.color_space))
           << ",\"pixel_range\":" << quote(enum_name(spec.images.pixel_range))
           << ",\"tensor_layout\":" << quote(enum_name(spec.images.tensor_layout))
           << ",\"resample_boundary\":" << quote(enum_name(spec.images.resample_boundary))
           << "},\"state\":{\"fields\":";
    string_array(output, spec.state.fields);
    output << ",\"real_dim\":" << spec.state.real_dim
           << ",\"model_dim\":" << spec.state.model_dim
           << ",\"pad_value\":" << spec.state.pad_value
           << ",\"normalization\":";
    normalization_json(output, spec.state.normalization, spec.state.stats);
    output << "},\"language\":{\"input_mode\":"
           << quote(enum_name(spec.language.input_mode))
           << ",\"prompt_template\":" << quote(spec.language.prompt_template)
           << ",\"tokenizer_family\":" << quote(spec.language.tokenizer_family)
           << ",\"tokenizer_revision\":" << quote(spec.language.tokenizer_revision)
           << ",\"max_tokens\":" << spec.language.max_tokens
           << ",\"text_encoder_in_artifact\":"
           << (spec.language.text_encoder_in_artifact ? "true" : "false")
           << ",\"padding_side\":" << quote(enum_name(spec.language.padding_side))
           << ",\"truncation_side\":" << quote(enum_name(spec.language.truncation_side))
           << ",\"attention_mask_required\":"
           << (spec.language.attention_mask_required ? "true" : "false")
           << ",\"special_token_ids\":";
    numeric_array(output, spec.language.special_token_ids);
    output << "},\"action\":{\"horizon\":" << spec.action.horizon
           << ",\"real_dim\":" << spec.action.real_dim
           << ",\"model_dim\":" << spec.action.model_dim
           << ",\"fields\":";
    string_array(output, spec.action.fields);
    output << ",\"representation\":" << quote(enum_name(spec.action.representation))
           << ",\"frame\":" << quote(enum_name(spec.action.frame))
           << ",\"gripper\":" << quote(enum_name(spec.action.gripper))
           << ",\"normalization\":";
    normalization_json(output, spec.action.normalization, spec.action.stats);
    output << ",\"recovery\":{\"kind\":"
           << quote(enum_name(spec.action.recovery.kind))
           << ",\"reference_state_indices\":";
    numeric_array(output, spec.action.recovery.reference_state_indices);
    output << "}}}";
}

} // namespace

std::string model_metadata_json(const ModelInfo & info,
                                const PolicySpec & spec) {
    std::ostringstream output;
    output << "{\"runtime_version\":{\"major\":" << WAM_VERSION_MAJOR
           << ",\"minor\":" << WAM_VERSION_MINOR
           << ",\"patch\":" << WAM_VERSION_PATCH
           << "},\"protocol_version\":{\"major\":0,\"minor\":5}"
           << ",\"architecture\":" << quote(info.architecture)
           << ",\"artifact_policy\":" << quote(spec.identity.profile)
           << ",\"artifact_sha256\":\"\""
           << ",\"artifact_bytes\":" << info.artifact_bytes
           << ",\"backend\":" << quote(backend_name(info.backend))
           << ",\"compute_precision\":" << quote(precision_name(info.compute_precision))
           << ",\"language_mode\":" << quote(language_runtime_name(info.language_mode))
           << ",\"capabilities\":{\"action\":"
           << (info.capabilities.action ? "true" : "false")
           << ",\"raw_images\":" << (info.capabilities.raw_images ? "true" : "false")
           << ",\"token_input\":" << (info.capabilities.token_input ? "true" : "false")
           << ",\"precomputed_embedding\":"
           << (info.capabilities.precomputed_embedding ? "true" : "false")
           << ",\"explicit_action_noise\":"
           << (info.capabilities.explicit_action_noise ? "true" : "false")
           << ",\"concurrent_sessions\":"
           << (info.capabilities.concurrent_sessions ? "true" : "false")
           << "},\"policy_spec\":";
    policy_json(output, spec);
    output << '}';
    return output.str();
}

std::string error_details_json(const std::vector<ErrorDetail> & details) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < details.size(); ++index) {
        if (index != 0) output << ',';
        output << "{\"field\":" << quote(details[index].field)
               << ",\"reason\":" << quote(details[index].reason) << '}';
    }
    output << ']';
    return output.str();
}

} // namespace wam::bindings
