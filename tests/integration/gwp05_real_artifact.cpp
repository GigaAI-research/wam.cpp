#include "models/common/gguf_reader.h"
#include "models/gwp05/artifact.h"
#include "policy/policy_spec.h"
#include "support/test_utils.h"

#include "wam/wam.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace {

using wam::test::require;

void require_tensor(const wam::internal::GgufReader & reader,
                    const std::string & name, wam::DType dtype) {
    const wam::internal::GgufTensorInfo & tensor =
        reader.require_tensor(name);
    require(tensor.dtype == dtype, name + " has an unexpected dtype");
    require(tensor.elements > 0, name + " is empty");
    require(tensor.bytes > 0, name + " has no stored payload");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        throw std::runtime_error(
            "usage: wam_gwp05_real_artifact /absolute/path/model.gguf");
    }

    const std::filesystem::path path =
        std::filesystem::absolute(argv[1]);
    require(std::filesystem::is_regular_file(path),
            "real GWP05 artifact does not exist");

    std::shared_ptr<wam::internal::GgufReader> reader =
        wam::internal::GgufReader::open(path.string());
    require(reader->require_string("general.architecture") == "gwp05",
            "real artifact architecture changed");
    require(reader->file_size() > UINT64_C(20) * 1024 * 1024 * 1024,
            "the external gate was given a synthetic or truncated artifact");
    const std::string conversion =
        reader->optional_string("gwp05.conversion_policy", "legacy-source");
    const bool packed_qkv = conversion == "mot-vae-bf16-qkv-v1";
    const bool mot_bf16 = conversion != "legacy-source" &&
        conversion != "source-f32-v1";
    const bool vae_bf16 = conversion == "mot-vae-bf16-v1" || packed_qkv;
    require(reader->tensor_count() == (packed_qkv ? 1756 : 1996),
            "real artifact tensor count differs from its conversion policy");

    require_tensor(*reader, "gwp.patch.weight",
                   mot_bf16 ? wam::DType::bf16 : wam::DType::f32);
    require_tensor(
        *reader,
        packed_qkv ? "gwp.blk.0.a.sa.qkv.weight"
                   : "gwp.blk.0.a.sa.q.weight",
        mot_bf16 ? wam::DType::bf16 : wam::DType::f32);
    require_tensor(*reader, "t5.token_embd.weight", wam::DType::bf16);
    require_tensor(*reader, "vae.enc.in.weight",
                   vae_bf16 ? wam::DType::bf16 : wam::DType::f32);
    require_tensor(*reader, "state_q01", wam::DType::f32);
    require_tensor(*reader, "action_q99", wam::DType::f32);

    const wam::internal::policy::PolicySpecDraft spec =
        wam::internal::gwp05::read_legacy_policy_spec(*reader);
    require(spec.identity.profile ==
                "legacy-gwp05-dual-arm-32d-quantile",
            "legacy GWP profile changed");
    require(spec.images.views.size() == 3,
            "legacy GWP view count changed");
    require(spec.images.composition.height == 384 &&
                spec.images.composition.width == 320 &&
                spec.images.resample_boundary ==
                    wam::internal::policy::ResampleBoundaryMode::clamp,
            "legacy GWP canvas changed");
    require(spec.state.real_dim == 14 && spec.state.model_dim == 32,
            "legacy GWP state contract changed");
    require(spec.action.horizon == 48 && spec.action.real_dim == 14 &&
                spec.action.model_dim == 32,
            "legacy GWP action contract changed");
    require(spec.state.normalization.kind ==
                wam::internal::policy::NormalizationKind::quantile &&
                spec.action.normalization.kind ==
                    wam::internal::policy::NormalizationKind::quantile,
            "legacy GWP normalization changed");
    require(spec.action.recovery.kind ==
                wam::internal::policy::ActionRecoveryKind::add_current_state,
            "legacy GWP action recovery changed");

    const std::shared_ptr<const wam::internal::gwp05::ArtifactContract>
        artifact = wam::internal::gwp05::load_artifact(reader, spec);
    require(artifact->geometry.hidden == 3072 &&
                artifact->geometry.layers == 30 &&
                artifact->geometry.heads == 24 &&
                artifact->geometry.head_dim == 128,
            "real GWP MoT geometry changed");
    require(artifact->geometry.action_hidden == 1024 &&
                artifact->geometry.inference_steps == 10 &&
                artifact->geometry.t5_hidden == 4096 &&
                artifact->geometry.t5_layers == 24 &&
                artifact->geometry.vae_z_dim == 48,
            "real GWP component geometry changed");
    require(artifact->sequence_geometry.visual_tokens == 120 &&
                artifact->sequence_geometry.action_tokens == 48,
            "real GWP token geometry changed");

    wam::ModelOptions options;
    options.artifact_path = path.string();
    options.backend = wam::Backend::cpu_metadata;
    options.language_mode =
        wam::LanguageRuntimeMode::external_embedding;
    wam::Model * model = wam::model_load(options);
    const wam::ModelInfo & info = wam::model_info(model);
    require(info.architecture == "gwp05" &&
                info.artifact_bytes == reader->file_size(),
            "public model metadata differs from the GGUF reader");
    require(!info.capabilities.action &&
                info.capabilities.explicit_action_noise,
            "Slice 3 capability boundary changed");
    wam::test::require_error(
        [&] { (void) wam::session_create(model); },
        wam::ErrorCode::unsupported,
        "Slice 4A must remain metadata-only");
    wam::model_free(model);
    return 0;
}
