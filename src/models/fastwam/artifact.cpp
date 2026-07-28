#include "models/fastwam/artifact.h"

#include "models/common/gguf_reader.h"
#include "wam/error.h"

#include "ggml.h"

#include <array>
#include <string>
#include <utility>

namespace wam::internal::fastwam {
namespace {

constexpr const char * kConversionPolicy = "fastwam-bf16-policy-v2";
constexpr const char * kConverterRevision =
    "wam-0.5-fastwam-policy-spec-v2";

[[noreturn]] void incompatible(const std::string & message,
                               const std::string & field,
                               const std::string & reason) {
    throw Error(ErrorCode::incompatible_artifact, message, {{field, reason}});
}

std::uint32_t positive_u32(const GgufReader & reader,
                           const std::string & key) {
    const std::uint32_t value = reader.require_u32("fastwam." + key);
    if (value == 0) {
        incompatible("FastWAM metadata contains zero geometry",
                     "fastwam." + key, "must be positive");
    }
    return value;
}

void require_bf16(const GgufReader & reader, const std::string & name) {
    if (reader.require_tensor(name).dtype != DType::bf16) {
        incompatible("FastWAM model tensor must be BF16", name,
                     "expected BF16 storage");
    }
}

std::vector<float> read_bf16(const GgufReader & reader,
                             const std::string & name) {
    const GgufTensorInfo & info = reader.require_tensor(name);
    if (info.dtype != DType::bf16) {
        incompatible("FastWAM tensor must be BF16", name,
                     "expected BF16 storage");
    }
    std::vector<ggml_bf16_t> source(info.elements);
    if (!reader.read_tensor(name, source.data(), source.size() * sizeof(source[0]))) {
        incompatible("cannot read FastWAM tensor", name, "I/O failure");
    }
    std::vector<float> result(source.size());
    ggml_bf16_to_fp32_row(source.data(), result.data(),
                          static_cast<std::int64_t>(source.size()));
    return result;
}

void require_transformer(const GgufReader & reader,
                         const std::string & component,
                         std::uint32_t layers) {
    for (const char * suffix : {
             "text.0.weight", "text.0.bias", "text.2.weight", "text.2.bias",
             "time.0.weight", "time.0.bias", "time.2.weight", "time.2.bias",
             "timep.1.weight", "timep.1.bias"}) {
        require_bf16(reader, "fastwam." + component + "." + suffix);
    }
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "fastwam." + component + ".blk." +
            std::to_string(layer) + ".";
        for (const char * suffix : {
                 "mod", "sa.q.weight", "sa.q.bias", "sa.k.weight",
                 "sa.k.bias", "sa.v.weight", "sa.v.bias", "sa.o.weight",
                 "sa.o.bias", "sa.qn.weight", "sa.kn.weight", "norm3.weight",
                 "norm3.bias", "ca.q.weight", "ca.q.bias", "ca.k.weight",
                 "ca.k.bias", "ca.v.weight", "ca.v.bias", "ca.o.weight",
                 "ca.o.bias", "ca.qn.weight", "ca.kn.weight", "ff.0.weight",
                 "ff.0.bias", "ff.2.weight", "ff.2.bias"}) {
            require_bf16(reader, prefix + suffix);
        }
    }
}

void require_vae_residual(const GgufReader & reader,
                          const std::string & prefix, bool shortcut) {
    for (const char * suffix : {
             "res.0.gamma", "res.2.weight", "res.2.bias", "res.3.gamma",
             "res.6.weight", "res.6.bias"}) {
        require_bf16(reader, prefix + "." + suffix);
    }
    if (shortcut) {
        require_bf16(reader, prefix + ".skip.weight");
        require_bf16(reader, prefix + ".skip.bias");
    }
}

void require_vae(const GgufReader & reader) {
    require_bf16(reader, "fastwam.vae.encoder.in.weight");
    require_bf16(reader, "fastwam.vae.encoder.in.bias");
    for (int block = 0; block < 4; ++block) {
        const std::string prefix = "fastwam.vae.encoder.down." +
            std::to_string(block) + ".downsamples.";
        require_vae_residual(reader, prefix + "0", block == 1 || block == 2);
        require_vae_residual(reader, prefix + "1", false);
        if (block != 3) {
            require_bf16(reader, prefix + "2.rs.1.weight");
            require_bf16(reader, prefix + "2.rs.1.bias");
        }
    }
    require_vae_residual(reader, "fastwam.vae.encoder.mid.0", false);
    require_vae_residual(reader, "fastwam.vae.encoder.mid.2", false);
    for (const char * name : {
             "fastwam.vae.encoder.mid.1.norm.gamma",
             "fastwam.vae.encoder.mid.1.qkv.weight",
             "fastwam.vae.encoder.mid.1.qkv.bias",
             "fastwam.vae.encoder.mid.1.p.weight",
             "fastwam.vae.encoder.mid.1.p.bias",
             "fastwam.vae.encoder.out.0.gamma",
             "fastwam.vae.encoder.out.2.weight",
             "fastwam.vae.encoder.out.2.bias",
             "fastwam.vae.quant.weight", "fastwam.vae.quant.bias"}) {
        require_bf16(reader, name);
    }
}

ArtifactComponentInfo component_info(const GgufReader & reader,
                                     const std::string & name,
                                     const std::string & prefix,
                                     DType dtype) {
    ArtifactComponentInfo result;
    result.name = name;
    result.source_dtype = dtype;
    result.stored_dtype = dtype;
    for (const GgufTensorInfo & tensor : reader.tensors()) {
        if (tensor.name.rfind(prefix, 0) != 0) continue;
        ++result.tensor_count;
        result.elements += tensor.elements;
        result.source_bytes += tensor.bytes;
        result.stored_bytes += tensor.bytes;
    }
    return result;
}

void require_component_count(const ArtifactComponentInfo & component,
                             std::uint64_t expected) {
    if (component.tensor_count != expected) {
        incompatible("FastWAM component tensor count is invalid",
                     "fastwam." + component.name,
                     "expected " + std::to_string(expected) + ", got " +
                         std::to_string(component.tensor_count));
    }
}

} // namespace

std::shared_ptr<const ArtifactContract> load_artifact(
    std::shared_ptr<GgufReader> reader,
    const policy::PolicySpec & policy_spec) {
    if (!reader) {
        incompatible("FastWAM artifact reader is null", "artifact",
                     "internal error");
    }
    if (reader->require_string("general.architecture") != "fastwam") {
        incompatible("artifact is not FastWAM", "general.architecture",
                     "expected fastwam");
    }

    auto artifact = std::make_shared<ArtifactContract>();
    artifact->reader = std::move(reader);
    artifact->conversion_policy =
        artifact->reader->require_string("fastwam.conversion_policy");
    if (artifact->conversion_policy != kConversionPolicy) {
        incompatible("unsupported FastWAM conversion policy",
                     "fastwam.conversion_policy",
                     artifact->conversion_policy);
    }
    if (artifact->reader->require_string("fastwam.converter_revision") !=
        kConverterRevision) {
        incompatible("unsupported FastWAM converter revision",
                     "fastwam.converter_revision", "mismatch");
    }

    Geometry & geometry = artifact->geometry;
    geometry.image_height = positive_u32(*artifact->reader, "image_height");
    geometry.image_width = positive_u32(*artifact->reader, "image_width");
    geometry.num_cameras = positive_u32(*artifact->reader, "num_cameras");
    geometry.action_dim = positive_u32(*artifact->reader, "action_dim");
    geometry.proprio_dim = positive_u32(*artifact->reader, "proprio_dim");
    geometry.action_horizon = positive_u32(*artifact->reader, "action_horizon");
    geometry.inference_steps = positive_u32(*artifact->reader, "inference_steps");
    geometry.latent_channels = positive_u32(*artifact->reader, "latent_channels");
    geometry.spatial_downsample = positive_u32(*artifact->reader, "spatial_downsample");
    geometry.temporal_downsample = positive_u32(*artifact->reader, "temporal_downsample");
    geometry.context_len = positive_u32(*artifact->reader, "context_len");
    geometry.text_dim = positive_u32(*artifact->reader, "text_dim");
    geometry.video_hidden_dim = positive_u32(*artifact->reader, "video_hidden_dim");
    geometry.action_hidden_dim = positive_u32(*artifact->reader, "action_hidden_dim");
    geometry.num_layers = positive_u32(*artifact->reader, "num_layers");
    geometry.num_heads = positive_u32(*artifact->reader, "num_heads");
    geometry.attn_head_dim = positive_u32(*artifact->reader, "attn_head_dim");
    geometry.action_shift = artifact->reader->require_f32("fastwam.action_shift");
    geometry.video_shift = artifact->reader->require_f32("fastwam.video_shift");
    geometry.norm_eps = artifact->reader->require_f32("fastwam.norm_eps");
    geometry.variant = artifact->reader->require_string("fastwam.variant");

    if (geometry.variant != "uncond_action_only") {
        incompatible("FastWAM variant is outside the 0.5 public scope",
                     "fastwam.variant", geometry.variant);
    }
    if (geometry.action_shift <= 0.0F || geometry.video_shift <= 0.0F ||
        geometry.norm_eps <= 0.0F) {
        incompatible("FastWAM scheduler and model norm values must be positive",
                     "fastwam.action_shift", "invalid value");
    }
    artifact->sequence_geometry = semantics::resolve_geometry({
        geometry.image_height, geometry.image_width,
        geometry.spatial_downsample, geometry.action_horizon,
        geometry.num_layers, geometry.num_heads, geometry.attn_head_dim,
        geometry.video_hidden_dim});

    for (const char * name : {
             "fastwam.video.patch.weight", "fastwam.video.patch.bias",
             "fastwam.action.encoder.weight", "fastwam.action.encoder.bias",
             "fastwam.action.head.weight", "fastwam.action.head.bias",
             "fastwam.proprio.weight", "fastwam.proprio.bias"}) {
        require_bf16(*artifact->reader, name);
    }
    require_transformer(*artifact->reader, "video", geometry.num_layers);
    require_transformer(*artifact->reader, "action", geometry.num_layers);
    require_vae(*artifact->reader);
    artifact->reader->require_shape(
        "fastwam.proprio.weight",
        {static_cast<std::int64_t>(geometry.proprio_dim),
         static_cast<std::int64_t>(geometry.text_dim)});
    artifact->reader->require_shape(
        "fastwam.proprio.bias",
        {static_cast<std::int64_t>(geometry.text_dim)});
    artifact->proprio_weight = read_bf16(
        *artifact->reader, "fastwam.proprio.weight");
    artifact->proprio_bias = read_bf16(
        *artifact->reader, "fastwam.proprio.bias");

    artifact->components = {
        component_info(*artifact->reader, "video", "fastwam.video.", DType::bf16),
        component_info(*artifact->reader, "action", "fastwam.action.", DType::bf16),
        component_info(*artifact->reader, "vae", "fastwam.vae.", DType::bf16),
        component_info(*artifact->reader, "proprio", "fastwam.proprio.", DType::bf16),
        component_info(*artifact->reader, "stats", "wam.norm.", DType::f32),
    };
    constexpr std::array<std::uint64_t, 5> expected_counts = {
        825, 824, 86, 2, 4};
    for (std::size_t index = 0; index < artifact->components.size(); ++index) {
        require_component_count(artifact->components[index],
                                expected_counts[index]);
    }

    validate_artifact(*artifact, policy_spec);
    return artifact;
}

void validate_artifact(const ArtifactContract & artifact,
                       const policy::PolicySpec & policy_spec) {
    semantics::validate_policy_semantics(policy_spec, artifact);
}

} // namespace wam::internal::fastwam
