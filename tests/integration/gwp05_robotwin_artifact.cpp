#include "artifact/artifact_view.h"
#include "models/gwp05/artifact.h"
#include "policy/policy_spec.h"
#include "support/test_utils.h"

#include "wam/wam.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

using wam::test::require;

void require_tensor(const wam::internal::GgufReader & reader,
                    const std::string & name, wam::DType dtype) {
    const wam::internal::GgufTensorInfo & tensor =
        reader.require_tensor(name);
    require(tensor.dtype == dtype, name + " has an unexpected dtype");
    require(tensor.elements > 0 && tensor.bytes > 0,
            name + " has an empty payload");
}

void require_positive_stats(const std::vector<float> & values,
                            const std::string & name) {
    require(values.size() == 14, name + " must contain 14 values");
    for (float value : values) {
        require(std::isfinite(value) && value > 1.0e-6F,
                name + " must be finite and positive");
    }
}

} // namespace

int main(int argc, char ** argv) {
    namespace policy = wam::internal::policy;
    namespace gwp05 = wam::internal::gwp05;

    if (argc != 2) {
        throw std::runtime_error(
            "usage: wam_gwp05_robotwin_artifact MODEL_GGUF");
    }
    const std::filesystem::path path =
        std::filesystem::absolute(argv[1]);
    require(std::filesystem::is_regular_file(path),
            "formal RoboTwin GGUF does not exist");

    const auto artifact_view =
        wam::internal::artifact::ArtifactView::open(path.string());
    std::shared_ptr<wam::internal::GgufReader> reader =
        artifact_view.shared_gguf();
    require(reader->file_size() > UINT64_C(20) * 1024 * 1024 * 1024,
            "formal RoboTwin GGUF is truncated or synthetic");
    require(reader->require_string("general.architecture") == "gwp05" &&
                reader->require_string("gwp05.architecture") == "gwp05",
            "formal RoboTwin GGUF has the wrong architecture");
    require(reader->require_string("gwp05.conversion_policy") ==
                "mot-vae-bf16-qkv-v1",
            "Gate A requires the native-BF16 packed-QKV artifact");
    require(reader->tensor_count() == 1756,
            "formal RoboTwin GGUF tensor count changed");
    require(reader->require_u64("gwp05.gwp_tensor_count") == 1424 &&
                reader->require_u64("gwp05.t5_tensor_count") == 242 &&
                reader->require_u64("gwp05.vae_tensor_count") == 82 &&
                reader->require_u64("gwp05.stats_tensor_count") == 8,
            "formal RoboTwin component telemetry changed");

    require_tensor(*reader, "gwp.patch.weight", wam::DType::bf16);
    require_tensor(*reader, "gwp.blk.0.a.sa.qkv.weight", wam::DType::bf16);
    require_tensor(*reader, "t5.token_embd.weight", wam::DType::bf16);
    require_tensor(*reader, "vae.enc.in.weight", wam::DType::bf16);
    for (const char * domain : {"state", "action"}) {
        for (const char * statistic : {"mean", "std", "q01", "q99"}) {
            require_tensor(*reader,
                           std::string("wam.norm.") + domain + "." + statistic,
                           wam::DType::f32);
        }
    }

    const std::optional<policy::PolicySpec> parsed =
        policy::try_read_policy_spec(artifact_view);
    require(parsed.has_value(), "formal RoboTwin GGUF has no PolicySpec");
    const policy::PolicySpec & spec = *parsed;
    require(spec.identity.artifact_schema_version == 2 &&
                spec.identity.profile ==
                    "gwp05_robotwin_dual_arm_14d_zscore" &&
                spec.identity.checkpoint_revision ==
                    "checkpoint_epoch_9_step_100000" &&
                spec.identity.training_dataset == "robotwin3_all" &&
                spec.identity.embodiment == "aloha_dual_arm",
            "formal RoboTwin PolicySpec identity changed");
    require(spec.images.views.size() == 3 &&
                spec.images.views[0].role == "camera_high" &&
                spec.images.views[1].role == "camera_left_wrist" &&
                spec.images.views[2].role == "camera_right_wrist" &&
                spec.images.composition.height == 384 &&
                spec.images.composition.width == 320 &&
                spec.images.resample_boundary ==
                    policy::ResampleBoundaryMode::truncate,
            "formal RoboTwin image contract changed");
    require(spec.state.real_dim == 14 && spec.state.model_dim == 14 &&
                spec.action.horizon == 48 && spec.action.real_dim == 14 &&
                spec.action.model_dim == 14,
            "formal RoboTwin state/action dimensions changed");
    require(spec.state.normalization.kind ==
                policy::NormalizationKind::z_score &&
                spec.action.normalization.kind ==
                    policy::NormalizationKind::z_score &&
                !spec.state.normalization.clip &&
                !spec.action.normalization.clip,
            "formal RoboTwin z-score contract changed");
    require_positive_stats(spec.state.stats.stddev, "state std");
    require_positive_stats(spec.action.stats.stddev, "action std");
    require(spec.action.recovery.kind ==
                policy::ActionRecoveryKind::add_current_state &&
                spec.action.recovery.reference_state_indices ==
                    std::vector<std::int32_t>(
                        {0, 1, 2, 3, 4, 5, -1,
                         7, 8, 9, 10, 11, 12, -1}),
            "formal RoboTwin action recovery changed");

    const std::shared_ptr<const gwp05::ArtifactContract> artifact =
        gwp05::load_artifact(reader, spec);
    require(!artifact->legacy_policy_spec &&
                artifact->geometry.action_dim == 14 &&
                artifact->geometry.real_state_dim == 14 &&
                artifact->geometry.real_action_dim == 14 &&
                artifact->sequence_geometry.action_tokens == 48,
            "formal RoboTwin private geometry changed");

    wam::RuntimeConfig options;
    options.backend = wam::Backend::cpu_metadata;
    options.language_mode = wam::LanguageRuntimeMode::external_embedding;
    wam::Model model = wam::Model::load(path.string(), options);
    const wam::ModelInfo & info = model.info();
    require(info.architecture == "gwp05" &&
                info.policy_spec != nullptr &&
                info.policy_spec->identity.profile == spec.identity.profile &&
                info.artifact_bytes == reader->file_size() &&
                info.capabilities.explicit_action_noise &&
                !info.capabilities.action,
            "formal RoboTwin public metadata contract changed");
    return 0;
}
