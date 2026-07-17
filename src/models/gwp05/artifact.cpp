#include "artifact.h"

#include "models/common/gguf_reader.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace wam::internal::gwp05 {
namespace {

[[noreturn]] void incompatible(const std::string & message,
                               const std::string & field,
                               const std::string & reason) {
    throw Error(ErrorCode::incompatible_artifact, message, {{field, reason}});
}

Geometry read_geometry(const GgufReader & reader) {
    Geometry value;
    // MoT
    value.hidden = reader.require_u32("gwp05.hidden");
    value.layers = reader.require_u32("gwp05.n_layers");
    value.heads = reader.require_u32("gwp05.n_heads");
    value.head_dim = reader.require_u32("gwp05.head_dim");
    value.ffn_dim = reader.require_u32("gwp05.ffn_dim");
    // Action expert
    value.action_hidden = reader.require_u32("gwp05.action_hidden");
    value.action_ffn_dim = reader.require_u32("gwp05.action_ffn_dim");
    value.action_dim = reader.require_u32("gwp05.action_dim");
    // State/action
    value.real_state_dim = reader.require_u32("gwp05.real_state_dim");
    value.real_action_dim = reader.require_u32("gwp05.real_action_dim");
    value.num_embodiments = reader.require_u32("gwp05.num_embodiments");
    value.embodiment_id = reader.require_u32("gwp05.embodiment_id");
    // Image
    value.image_height = reader.require_u32("gwp05.image_height");
    value.image_width = reader.require_u32("gwp05.image_width");
    value.num_views = reader.require_u32("gwp05.num_views");
    // Inference
    value.action_chunk = reader.require_u32("gwp05.action_chunk");
    value.inference_steps = reader.require_u32("gwp05.inference_steps");
    value.flow_shift = reader.require_f32("gwp05.flow_shift");
    // Normalization
    value.norm_eps = reader.require_f32("gwp05.norm_eps");
    // T5
    value.t5_vocab_size = reader.require_u32("gwp05.t5_vocab_size");
    value.t5_hidden = reader.require_u32("gwp05.t5_hidden");
    value.t5_ffn_dim = reader.require_u32("gwp05.t5_ffn_dim");
    value.t5_heads = reader.require_u32("gwp05.t5_heads");
    value.t5_head_dim = reader.require_u32("gwp05.t5_head_dim");
    value.t5_layers = reader.require_u32("gwp05.t5_layers");
    value.t5_max_length = reader.has("gwp05.t5_max_length")
        ? reader.require_u32("gwp05.t5_max_length") : 512;
    // VAE
    value.vae_z_dim = reader.require_u32("gwp05.vae_z_dim");

    const bool zero = value.hidden == 0 || value.layers == 0 || value.heads == 0 ||
        value.head_dim == 0 || value.ffn_dim == 0 || value.action_hidden == 0 ||
        value.action_ffn_dim == 0 || value.action_dim == 0 || value.num_embodiments == 0 ||
        value.image_height == 0 || value.image_width == 0 || value.action_chunk == 0 ||
        value.inference_steps == 0 || value.t5_vocab_size == 0 || value.t5_hidden == 0 ||
        value.t5_ffn_dim == 0 || value.t5_heads == 0 || value.t5_head_dim == 0 ||
        value.t5_layers == 0 || value.vae_z_dim == 0;
    if (zero) incompatible("GWP geometry contains a zero dimension", "gwp05", "zero dimension");
    if (value.hidden != value.heads * value.head_dim ||
        value.t5_hidden != value.t5_heads * value.t5_head_dim) {
        incompatible("GWP attention geometry is inconsistent", "gwp05.head_dim",
                     "hidden size must equal heads multiplied by head_dim");
    }
    if (value.real_state_dim > value.action_dim || value.real_action_dim > value.action_dim) {
        incompatible("GWP action geometry is inconsistent", "gwp05.action_dim",
                     "real dimensions exceed padded action dimension");
    }
    if (value.num_views != 3) {
        incompatible("GWP-0.5 requires three camera views", "gwp05.num_views",
                     std::to_string(value.num_views));
    }
    if (value.embodiment_id >= value.num_embodiments) {
        incompatible("GWP embodiment id is out of range", "gwp05.embodiment_id",
                     std::to_string(value.embodiment_id));
    }
    if (value.image_height % 32 != 0 || value.image_width % 32 != 0) {
        incompatible("GWP image geometry must be divisible by 32", "gwp05.image_height",
                     "invalid image geometry");
    }
    if (!std::isfinite(value.flow_shift) || value.flow_shift <= 0.0f ||
        !std::isfinite(value.norm_eps) || value.norm_eps <= 0.0f) {
        incompatible("GWP floating-point metadata is invalid", "gwp05.flow_shift",
                     "values must be finite and positive");
    }
    return value;
}

void validate_schema(const GgufReader & reader, const Geometry & geometry,
                     bool packed_qkv) {
    if (reader.require_string("gwp05.architecture") != "gwp05") {
        incompatible("wrong GWP architecture marker", "gwp05.architecture", "expected gwp05");
    }
    reader.require_shape("gwp.patch.weight", {2, 2, geometry.vae_z_dim, geometry.hidden});
    reader.require_shape("gwp.state.in.weight",
                         {128, geometry.action_dim, geometry.num_embodiments});
    reader.require_shape("gwp.decode.out.weight",
                         {geometry.action_dim, 128, geometry.num_embodiments});
    reader.require_shape("t5.token_embd.weight", {geometry.t5_hidden, geometry.t5_vocab_size});
    reader.require_shape("vae.enc.in.weight", {3, 3, 12, 160});
    reader.require_shape("state_q01", {geometry.action_dim});
    reader.require_shape("action_q99", {geometry.action_dim});

    for (std::uint32_t layer = 0; layer < geometry.layers; ++layer) {
        const std::string base = "gwp.blk." + std::to_string(layer);
        const std::string projection = packed_qkv ? ".sa.qkv.weight" : ".sa.q.weight";
        const std::int64_t output = packed_qkv ? 3LL * geometry.hidden : geometry.hidden;
        reader.require_shape(base + ".a" + projection, {geometry.action_hidden, output});
        reader.require_shape(base + ".v" + projection, {geometry.hidden, output});
    }
}

bool is_statistic(const std::string & name) {
    static constexpr std::array<const char *, 8> names = {
        "state_mean", "state_std", "state_q01", "state_q99",
        "action_mean", "action_std", "action_q01", "action_q99",
    };
    for (const char * expected : names) if (name == expected) return true;
    return false;
}

DType dtype(ggml_type type) {
    if (type == GGML_TYPE_F32) return DType::f32;
    if (type == GGML_TYPE_BF16) return DType::bf16;
    return DType::unknown;
}

const char * dtype_name(DType type) {
    if (type == DType::f32) return "F32";
    if (type == DType::bf16) return "BF16";
    return "UNKNOWN";
}

struct ExpectedComponent {
    ArtifactComponentInfo telemetry;
    const char * prefix = nullptr;
    std::uint64_t expected_count = 0;
};

std::vector<ArtifactComponentInfo> validate_components(
    const GgufReader & reader, const Geometry & geometry,
    const std::string & policy, bool explicit_policy) {
    const bool packed = policy == "mot-vae-bf16-qkv-v1";
    const bool mot_bf16 = policy == "mot-bf16-v1" ||
        policy == "mot-vae-bf16-v1" || packed;
    const bool vae_bf16 = policy == "mot-vae-bf16-v1" || packed;
    std::array<ExpectedComponent, 4> components = {{
        {{{"gwp"}, DType::f32, mot_bf16 ? DType::bf16 : DType::f32}, "gwp.",
         static_cast<std::uint64_t>(44 + (packed ? 46 : 54) * geometry.layers)},
        {{{"t5"}, DType::bf16, DType::bf16}, "t5.",
         static_cast<std::uint64_t>(2 + 10 * geometry.t5_layers)},
        {{{"vae"}, DType::f32, vae_bf16 ? DType::bf16 : DType::f32}, "vae.", 82},
        {{{"stats"}, DType::f32, DType::f32}, nullptr, 8},
    }};

    for (const GgufTensorInfo & tensor : reader.tensors()) {
        ExpectedComponent * component = nullptr;
        for (ExpectedComponent & candidate : components) {
            if ((candidate.prefix != nullptr &&
                 tensor.name.compare(0, std::strlen(candidate.prefix), candidate.prefix) == 0) ||
                (candidate.prefix == nullptr && is_statistic(tensor.name))) {
                component = &candidate;
                break;
            }
        }
        if (component == nullptr) {
            incompatible("GWP policy does not cover a tensor", tensor.name, "unknown component");
        }
        if (dtype(tensor.type) != component->telemetry.stored_dtype) {
            incompatible("GWP tensor dtype conflicts with artifact policy", tensor.name,
                         std::string("expected ") + dtype_name(component->telemetry.stored_dtype));
        }
        ArtifactComponentInfo & info = component->telemetry;
        ++info.tensor_count;
        if (tensor.elements > std::numeric_limits<std::uint64_t>::max() - info.elements ||
            tensor.bytes > std::numeric_limits<std::uint64_t>::max() - info.stored_bytes) {
            incompatible("GWP component telemetry overflows", info.name, "overflow");
        }
        info.elements += tensor.elements;
        info.stored_bytes += tensor.bytes;
    }

    std::vector<ArtifactComponentInfo> result;
    result.reserve(components.size());
    for (ExpectedComponent & component : components) {
        ArtifactComponentInfo & info = component.telemetry;
        const std::uint64_t source_width = dtype_size(info.source_dtype);
        if (info.elements > std::numeric_limits<std::uint64_t>::max() / source_width) {
            incompatible("GWP source byte telemetry overflows", info.name, "overflow");
        }
        info.source_bytes = info.elements * source_width;
        if (info.tensor_count != component.expected_count) {
            incompatible("GWP component tensor count is invalid", info.name,
                         "expected " + std::to_string(component.expected_count) +
                         ", got " + std::to_string(info.tensor_count));
        }
        if (explicit_policy) {
            const std::string base = "gwp05." + info.name + "_";
            if (reader.require_string(base + "source_dtype") != dtype_name(info.source_dtype) ||
                reader.require_string(base + "stored_dtype") != dtype_name(info.stored_dtype) ||
                reader.require_u64(base + "tensor_count") != info.tensor_count ||
                reader.require_u64(base + "elements") != info.elements ||
                reader.require_u64(base + "source_bytes") != info.source_bytes ||
                reader.require_u64(base + "stored_bytes") != info.stored_bytes) {
                incompatible("GWP component metadata does not match tensor descriptors", info.name,
                             "count, dtype, element, or byte mismatch");
            }
        }
        result.push_back(info);
    }
    return result;
}

void validate_requested_precision(const ModelOptions & options, const std::string & policy) {
    const bool artifact_bf16 = policy == "mot-bf16-v1" ||
        policy == "mot-vae-bf16-v1" || policy == "mot-vae-bf16-qkv-v1";
    const Precision required = artifact_bf16 ? Precision::bf16_latency : Precision::f32_reference;
    if (options.precision != required) {
        throw Error(ErrorCode::unsupported, "requested precision conflicts with artifact policy",
                    {{"precision", artifact_bf16 ? "bf16_latency required" :
                                                   "f32_reference required"},
                     {"gwp05.conversion_policy", policy}});
    }
    if (options.backend != Backend::automatic && options.backend != Backend::cpu_metadata &&
        options.backend != Backend::cuda) {
        throw Error(ErrorCode::unsupported, "unsupported backend for GWP artifact metadata",
                    {{"backend", std::to_string(static_cast<std::uint32_t>(options.backend))}});
    }
}

} // namespace

ArtifactContract inspect_artifact(std::shared_ptr<GgufReader> reader,
                                  const ModelOptions & options) {
    ArtifactContract artifact;
    artifact.reader = std::move(reader);
    artifact.geometry = read_geometry(*artifact.reader);
    try {
        artifact.sequence_geometry = semantics::resolve_geometry({
            artifact.geometry.image_height,
            artifact.geometry.image_width,
            artifact.geometry.num_views,
            artifact.geometry.action_chunk,
            artifact.geometry.layers,
            artifact.geometry.inference_steps,
            artifact.geometry.hidden,
            artifact.geometry.action_hidden,
            artifact.geometry.heads,
            artifact.geometry.heads,
            artifact.geometry.head_dim,
            artifact.geometry.t5_max_length,
        });
    } catch (const Error & error) {
        if (error.code() != ErrorCode::invalid_argument) throw;
        throw Error(ErrorCode::incompatible_artifact,
                    "GWP artifact has invalid structural semantics", error.details());
    }

    const std::string conversion = artifact.reader->optional_string("gwp05.conversion_policy");
    const bool explicit_policy = !conversion.empty();
    artifact.policy = explicit_policy ? conversion : "legacy-source";
    if (explicit_policy && conversion != "source-f32-v1" && conversion != "mot-bf16-v1" &&
        conversion != "mot-vae-bf16-v1" && conversion != "mot-vae-bf16-qkv-v1") {
        throw Error(ErrorCode::unsupported, "unsupported GWP conversion policy",
                    {{"gwp05.conversion_policy", conversion}});
    }
    if (!explicit_policy) {
        const std::string legacy = artifact.reader->optional_string("gwp05.weight_policy", "source");
        if (legacy != "source") {
            throw Error(ErrorCode::unsupported, "legacy GWP weight policy is unsupported",
                        {{"gwp05.weight_policy", legacy}});
        }
    } else if (artifact.reader->require_string("gwp05.converter_revision") !=
                   "gwp05-native-bf16-v1" ||
               artifact.reader->require_string("gwp05.weight_policy") != conversion) {
        incompatible("GWP conversion metadata is inconsistent", "gwp05.converter_revision",
                     "revision or weight policy mismatch");
    }

    validate_requested_precision(options, artifact.policy);
    validate_schema(*artifact.reader, artifact.geometry,
                    conversion == "mot-vae-bf16-qkv-v1");
    artifact.components = validate_components(*artifact.reader, artifact.geometry,
                                              artifact.policy, explicit_policy);
    return artifact;
}

} // namespace wam::internal::gwp05
