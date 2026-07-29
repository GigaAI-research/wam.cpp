#include "models/gwp05/contract.h"

#include "artifact/gguf_reader.h"
#include "artifact/tensor_spec.h"
#include "wam/error.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace wam::internal::gwp05 {
namespace {

[[noreturn]] void incompatible(const std::string & message,
                               const std::string & field,
                               const std::string & reason) {
    throw Error(ErrorCode::incompatible_artifact, message, {{field, reason}});
}

[[noreturn]] void unsupported(const std::string & message,
                              const std::string & field,
                              const std::string & reason) {
    throw Error(ErrorCode::unsupported, message, {{field, reason}});
}

Geometry read_geometry(const GgufReader & reader) {
    Geometry geometry;
    geometry.hidden = reader.require_u32("gwp05.hidden");
    geometry.layers = reader.require_u32("gwp05.n_layers");
    geometry.heads = reader.require_u32("gwp05.n_heads");
    geometry.head_dim = reader.require_u32("gwp05.head_dim");
    geometry.ffn_dim = reader.require_u32("gwp05.ffn_dim");
    geometry.action_hidden = reader.require_u32("gwp05.action_hidden");
    geometry.action_ffn_dim = reader.require_u32("gwp05.action_ffn_dim");
    geometry.num_embodiments = reader.require_u32("gwp05.num_embodiments");
    geometry.embodiment_id = reader.require_u32("gwp05.embodiment_id");
    geometry.inference_steps = reader.require_u32("gwp05.inference_steps");
    geometry.flow_shift = reader.require_f32("gwp05.flow_shift");
    geometry.norm_epsilon = reader.require_f32("gwp05.norm_eps");
    geometry.t5_vocab_size = reader.require_u32("gwp05.t5_vocab_size");
    geometry.t5_hidden = reader.require_u32("gwp05.t5_hidden");
    geometry.t5_ffn_dim = reader.require_u32("gwp05.t5_ffn_dim");
    geometry.t5_heads = reader.require_u32("gwp05.t5_heads");
    geometry.t5_head_dim = reader.require_u32("gwp05.t5_head_dim");
    geometry.t5_layers = reader.require_u32("gwp05.t5_layers");
    geometry.t5_max_length = reader.has("gwp05.t5_max_length")
        ? reader.require_u32("gwp05.t5_max_length")
        : 512;
    geometry.vae_z_dim = reader.require_u32("gwp05.vae_z_dim");

    const bool zero = geometry.hidden == 0 || geometry.layers == 0 ||
        geometry.heads == 0 || geometry.head_dim == 0 ||
        geometry.ffn_dim == 0 || geometry.action_hidden == 0 ||
        geometry.action_ffn_dim == 0 || geometry.num_embodiments == 0 ||
        geometry.inference_steps == 0 || geometry.t5_vocab_size == 0 ||
        geometry.t5_hidden == 0 || geometry.t5_ffn_dim == 0 ||
        geometry.t5_heads == 0 || geometry.t5_head_dim == 0 ||
        geometry.t5_layers == 0 || geometry.t5_max_length == 0 ||
        geometry.vae_z_dim == 0;
    if (zero) {
        incompatible("GWP geometry contains a zero dimension", "gwp05",
                     "zero dimension");
    }
    if (geometry.hidden !=
            static_cast<std::uint64_t>(geometry.heads) * geometry.head_dim ||
        geometry.t5_hidden !=
            static_cast<std::uint64_t>(geometry.t5_heads) *
                geometry.t5_head_dim) {
        incompatible("GWP attention geometry is inconsistent",
                     "gwp05.head_dim",
                     "hidden size must equal heads multiplied by head_dim");
    }
    if (geometry.embodiment_id >= geometry.num_embodiments) {
        incompatible("GWP embodiment id is out of range",
                     "gwp05.embodiment_id",
                     std::to_string(geometry.embodiment_id));
    }
    if (!std::isfinite(geometry.flow_shift) ||
        geometry.flow_shift <= 0.0F ||
        !std::isfinite(geometry.norm_epsilon) ||
        geometry.norm_epsilon <= 0.0F) {
        incompatible("GWP floating-point geometry is invalid",
                     "gwp05.flow_shift",
                     "flow_shift and norm_eps must be finite and positive");
    }
    return geometry;
}

void cross_check_u32(const GgufReader & reader, const std::string & key,
                     std::size_t expected) {
    if (!reader.has(key)) return;
    const std::uint32_t value = reader.require_u32(key);
    if (value != expected) {
        incompatible("legacy GWP metadata conflicts with PolicySpec", key,
                     "expected " + std::to_string(expected) + ", got " +
                         std::to_string(value));
    }
}

void cross_check_stat(const GgufReader & reader, const std::string & legacy,
                      const std::vector<float> & expected) {
    if (reader.find_tensor(legacy) == nullptr) return;
    reader.require_shape(legacy,
                         {static_cast<std::int64_t>(expected.size())});
    const std::vector<float> actual = reader.read_f32_tensor(legacy);
    if (actual != expected) {
        incompatible("legacy normalization tensor conflicts with PolicySpec",
                     legacy, "payload mismatch");
    }
}

void validate_policy_stat_tensors(const GgufReader & reader,
                                  const policy::PolicySpec & policy_spec) {
    const auto validate_domain = [&](const std::string & domain,
                                     std::size_t dimension) {
        const std::vector<std::int64_t> shape = {
            static_cast<std::int64_t>(dimension)};
        for (const char * statistic : {"mean", "std", "lower", "upper"}) {
            const std::string name =
                "wam.norm." + domain + "." + statistic;
            if (reader.find_tensor(name) != nullptr) {
                artifact::validate_tensor(
                    reader, {name, {DType::f32}, shape, true});
            }
        }
    };
    validate_domain("state", policy_spec.state.model_dim);
    validate_domain("action", policy_spec.action.model_dim);
}

std::string read_conversion_policy(const GgufReader & reader) {
    const std::string conversion =
        reader.optional_string("gwp05.conversion_policy");
    if (conversion.empty()) {
        const std::string legacy =
            reader.optional_string("gwp05.weight_policy", "source");
        if (legacy != "source") {
            unsupported("legacy GWP weight policy is unsupported",
                        "gwp05.weight_policy", legacy);
        }
        return "legacy-source";
    }
    if (conversion != "source-f32-v1" && conversion != "mot-bf16-v1" &&
        conversion != "mot-vae-bf16-v1" &&
        conversion != "mot-vae-bf16-qkv-v1") {
        unsupported("unsupported GWP conversion policy",
                    "gwp05.conversion_policy", conversion);
    }
    if (reader.has("gwp05.weight_policy") &&
        reader.require_string("gwp05.weight_policy") != conversion) {
        incompatible("GWP conversion metadata is inconsistent",
                     "gwp05.weight_policy", "does not match conversion_policy");
    }
    return conversion;
}

std::vector<std::string> legacy_fields() {
    return {
        "left.joint.0", "left.joint.1", "left.joint.2",
        "left.joint.3", "left.joint.4", "left.joint.5",
        "left.gripper", "right.joint.0", "right.joint.1",
        "right.joint.2", "right.joint.3", "right.joint.4",
        "right.joint.5", "right.gripper",
    };
}

policy::NormalizationStats read_legacy_stats(const GgufReader & reader,
                                             const std::string & domain,
                                             std::size_t model_dim,
                                             std::size_t real_dim) {
    policy::NormalizationStats stats;
    const std::string q01 = domain + "_q01";
    const std::string q99 = domain + "_q99";
    reader.require_shape(q01, {static_cast<std::int64_t>(model_dim)});
    reader.require_shape(q99, {static_cast<std::int64_t>(model_dim)});
    stats.lower = reader.read_f32_tensor(q01);
    stats.upper = reader.read_f32_tensor(q99);
    stats.mask.assign(model_dim, 0);
    std::fill_n(stats.mask.begin(), real_dim, 1);
    return stats;
}

} // namespace

policy::PolicySpec read_legacy_policy_spec(const GgufReader & reader) {
    if (reader.has("wam.artifact_schema_version")) {
        incompatible("legacy GWP migration received a schema artifact",
                     "wam.artifact_schema_version", "unexpected key");
    }
    if (reader.has("gwp05.architecture") &&
        reader.require_string("gwp05.architecture") != "gwp05") {
        incompatible("wrong GWP architecture marker", "gwp05.architecture",
                     "expected gwp05");
    }

    const std::size_t model_dim = reader.require_u32("gwp05.action_dim");
    const std::size_t state_dim = reader.require_u32("gwp05.real_state_dim");
    const std::size_t action_dim = reader.require_u32("gwp05.real_action_dim");
    const std::uint32_t image_height =
        reader.require_u32("gwp05.image_height");
    const std::uint32_t image_width =
        reader.require_u32("gwp05.image_width");
    const std::size_t views = reader.require_u32("gwp05.num_views");
    const std::size_t horizon = reader.require_u32("gwp05.action_chunk");
    if (views != 3 || state_dim != 14 || action_dim != 14 ||
        model_dim != 32 || horizon != 48 || image_height == 0 ||
        image_width == 0) {
        unsupported("legacy GWP metadata is not the audited 32D dual-arm contract",
                    "gwp05", "unsupported legacy geometry");
    }

    policy::PolicySpec spec;
    spec.identity.artifact_schema_version =
        policy::kPolicySpecSchemaVersion;
    spec.identity.profile = "legacy-gwp05-dual-arm-32d-quantile";
    spec.identity.checkpoint_revision = "legacy-unversioned";
    spec.identity.training_dataset = "legacy-unknown";
    spec.identity.embodiment = "dual_arm";

    const std::uint32_t top_height = image_height / 2;
    const std::uint32_t bottom_height = image_height - top_height;
    const std::uint32_t left_width = image_width / 2;
    const std::uint32_t right_width = image_width - left_width;
    const std::vector<std::string> roles = {
        "camera_high", "camera_left_wrist", "camera_right_wrist"};
    const std::vector<policy::ImagePlacement> placements = {
        {roles[0], 0, 0, image_width, top_height},
        {roles[1], 0, top_height, left_width, bottom_height},
        {roles[2], left_width, top_height, right_width, bottom_height},
    };
    for (const policy::ImagePlacement & placement : placements) {
        spec.images.views.push_back(
            {placement.role, placement.height, placement.width,
             policy::ResizeMode::cover_center_crop,
             policy::InterpolationMode::bilinear, true});
    }
    spec.images.composition.kind = policy::ImageCompositionKind::canvas;
    spec.images.composition.height = image_height;
    spec.images.composition.width = image_width;
    spec.images.composition.placements = placements;
    spec.images.color_space = policy::ColorSpace::rgb;
    spec.images.pixel_range = policy::PixelRange::minus_one_to_one;
    spec.images.tensor_layout = policy::TensorLayout::chw;
    spec.images.resample_boundary =
        policy::ResampleBoundaryMode::clamp;

    spec.state.fields = legacy_fields();
    spec.state.real_dim = state_dim;
    spec.state.model_dim = model_dim;
    spec.state.pad_value = 0.0F;
    spec.state.normalization = {policy::NormalizationKind::quantile, false,
                                1.0e-8F};
    spec.state.stats =
        read_legacy_stats(reader, "state", model_dim, state_dim);

    spec.language.input_mode = policy::LanguageInputMode::tokens_or_embedding;
    spec.language.prompt_template = "{task}";
    spec.language.tokenizer_family = "umt5";
    spec.language.tokenizer_revision = "legacy-artifact";
    spec.language.max_tokens = semantics::kPromptTokens;
    spec.language.text_encoder_in_artifact = true;
    spec.language.padding_side = policy::SequenceSide::right;
    spec.language.truncation_side = policy::SequenceSide::right;
    spec.language.attention_mask_required = true;

    spec.action.horizon = horizon;
    spec.action.real_dim = action_dim;
    spec.action.model_dim = model_dim;
    spec.action.fields = legacy_fields();
    spec.action.representation = policy::ActionRepresentation::joint_position;
    spec.action.frame = policy::ActionFrame::controller;
    spec.action.gripper = policy::GripperEncoding::continuous;
    spec.action.normalization = {policy::NormalizationKind::quantile, false,
                                 1.0e-8F};
    spec.action.stats =
        read_legacy_stats(reader, "action", model_dim, action_dim);
    spec.action.recovery.kind = policy::ActionRecoveryKind::add_current_state;
    spec.action.recovery.reference_state_indices = {
        0, 1, 2, 3, 4, 5, -1, 7, 8, 9, 10, 11, 12, -1};

    policy::validate_policy_spec(spec);
    semantics::validate_policy_semantics(spec);
    return spec;
}

void validate_contract(const Gwp05Contract & artifact,
                       const policy::PolicySpec & policy_spec) {
    if (artifact.reader == nullptr) {
        throw Error(ErrorCode::internal,
                    "GWP artifact has no backing GGUF reader");
    }
    policy::validate_policy_spec(policy_spec);
    try {
        semantics::validate_policy_semantics(policy_spec);
    } catch (const Error & error) {
        if (error.code() != ErrorCode::invalid_argument) throw;
        throw Error(ErrorCode::incompatible_artifact,
                    "GWP PolicySpec has unsupported semantics",
                    error.details());
    }

    const GgufReader & reader = *artifact.reader;
    validate_policy_stat_tensors(reader, policy_spec);
    if (!artifact.legacy_policy_spec &&
        !reader.has("gwp05.t5_max_length")) {
        incompatible("GWP artifact is missing text capacity",
                     "gwp05.t5_max_length", "missing");
    }
    if (reader.has("gwp05.architecture") &&
        reader.require_string("gwp05.architecture") != "gwp05") {
        incompatible("wrong GWP architecture marker", "gwp05.architecture",
                     "expected gwp05");
    }
    cross_check_u32(reader, "gwp05.action_dim",
                    policy_spec.action.model_dim);
    cross_check_u32(reader, "gwp05.real_state_dim",
                    policy_spec.state.real_dim);
    cross_check_u32(reader, "gwp05.real_action_dim",
                    policy_spec.action.real_dim);
    cross_check_u32(reader, "gwp05.num_views",
                    policy_spec.images.views.size());
    cross_check_u32(reader, "gwp05.action_chunk",
                    policy_spec.action.horizon);
    cross_check_u32(reader, "gwp05.image_height",
                    policy_spec.images.composition.height);
    cross_check_u32(reader, "gwp05.image_width",
                    policy_spec.images.composition.width);

    cross_check_stat(reader, "state_q01", policy_spec.state.stats.lower);
    cross_check_stat(reader, "state_q99", policy_spec.state.stats.upper);
    cross_check_stat(reader, "action_q01", policy_spec.action.stats.lower);
    cross_check_stat(reader, "action_q99", policy_spec.action.stats.upper);

    if (artifact.geometry.t5_max_length < semantics::kPromptTokens ||
        artifact.geometry.t5_max_length < policy_spec.language.max_tokens) {
        incompatible("GWP text capacity is smaller than PolicySpec",
                     "gwp05.t5_max_length", "insufficient capacity");
    }
}

std::shared_ptr<const Gwp05Contract> load_contract(
    std::shared_ptr<GgufReader> reader,
    const policy::PolicySpec & policy_spec) {
    if (reader == nullptr) {
        throw Error(ErrorCode::internal,
                    "cannot load GWP artifact from a null GGUF reader");
    }
    auto artifact = std::make_shared<Gwp05Contract>();
    artifact->reader = std::move(reader);
    artifact->geometry = read_geometry(*artifact->reader);
    artifact->geometry.state_dim =
        static_cast<std::uint32_t>(policy_spec.state.model_dim);
    artifact->geometry.action_dim =
        static_cast<std::uint32_t>(policy_spec.action.model_dim);
    artifact->geometry.real_state_dim =
        static_cast<std::uint32_t>(policy_spec.state.real_dim);
    artifact->geometry.real_action_dim =
        static_cast<std::uint32_t>(policy_spec.action.real_dim);
    artifact->geometry.image_height =
        policy_spec.images.composition.height;
    artifact->geometry.image_width =
        policy_spec.images.composition.width;
    artifact->geometry.num_views =
        static_cast<std::uint32_t>(policy_spec.images.views.size());
    artifact->geometry.action_chunk =
        static_cast<std::uint32_t>(policy_spec.action.horizon);
    artifact->conversion_policy = read_conversion_policy(*artifact->reader);
    artifact->vae_latents_mean =
        artifact->reader->optional_f32_array("gwp05.vae_latents_mean");
    artifact->vae_latents_std =
        artifact->reader->optional_f32_array("gwp05.vae_latents_std");
    artifact->legacy_policy_spec =
        !artifact->reader->has("wam.artifact_schema_version");
    validate_contract(*artifact, policy_spec);

    try {
        artifact->sequence_geometry = semantics::resolve_sequence_geometry({
            static_cast<std::int64_t>(
                policy_spec.images.composition.height),
            static_cast<std::int64_t>(
                policy_spec.images.composition.width),
            static_cast<std::int64_t>(policy_spec.action.horizon),
            artifact->geometry.layers,
            artifact->geometry.inference_steps,
            artifact->geometry.hidden,
            artifact->geometry.action_hidden,
            artifact->geometry.heads,
            artifact->geometry.heads,
            artifact->geometry.head_dim,
            artifact->geometry.t5_max_length,
        });
    } catch (const Error & error) {
        if (error.code() != ErrorCode::invalid_argument) throw;
        throw Error(ErrorCode::incompatible_artifact,
                    "GWP artifact has invalid structural geometry",
                    error.details());
    }
    return artifact;
}

void validate_runtime_contract(const Gwp05Contract & contract) {
    if (contract.vae_latents_mean.size() != contract.geometry.vae_z_dim ||
        contract.vae_latents_std.size() != contract.geometry.vae_z_dim) {
        incompatible("GWP VAE statistics have invalid dimensions",
                     "gwp05.vae_latents_mean/std", "size mismatch");
    }
    if (std::any_of(
            contract.vae_latents_std.begin(), contract.vae_latents_std.end(),
            [](float value) {
                return !(value > 0.0F) || !std::isfinite(value);
            })) {
        incompatible("GWP VAE latent standard deviation is invalid",
                     "gwp05.vae_latents_std", "non-positive or non-finite");
    }
}

} // namespace wam::internal::gwp05
