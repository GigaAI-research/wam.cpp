#include "fixtures/metadata_fixture.h"
#include "artifact/artifact_view.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "artifact/gguf_reader.h"
#include "models/gwp05/scheduler.h"
#include "models/gwp05/semantics.h"
#include "policy/policy_spec.h"

#include <cmath>
#include <limits>
#include <vector>

namespace semantics = wam::internal::gwp05::semantics;

int main() {
    using wam::test::require;
    using wam::test::require_error;

    const semantics::SequenceGeometry geometry =
        semantics::resolve_sequence_geometry(
            {384, 320, 48, 30, 10, 3072, 1024, 24, 24, 128, 512});
    require(geometry.latent_height == 24 && geometry.latent_width == 20,
            "GWP latent geometry changed");
    require(geometry.visual_grid_height == 12 &&
                geometry.visual_grid_width == 10 &&
                geometry.visual_tokens == 120,
            "GWP visual token geometry changed");
    require(geometry.prefix_tokens == 121 && geometry.action_tokens == 48 &&
                geometry.full_tokens == 169,
            "GWP sequence lengths changed");

    const semantics::TokenLayout layout =
        semantics::complete_mot_layout(geometry);
    require(layout.action_offset == 1 && layout.visual_offset == 49 &&
                layout.total_tokens == 169,
            "GWP complete-MoT layout changed");
    require(semantics::action_positions(3, true) ==
                std::vector<std::int32_t>({0, 1, 2, 3}),
            "GWP action positions changed");
    require(semantics::action_timesteps(2, 7.0F, true) ==
                std::vector<float>({0.0F, 7.0F, 7.0F}),
            "GWP action timesteps changed");
    const semantics::VisualPositions positions =
        semantics::visual_positions(2, 3);
    require(positions.height ==
                std::vector<std::int32_t>({0, 0, 0, 1, 1, 1}) &&
                positions.width ==
                    std::vector<std::int32_t>({0, 1, 2, 0, 1, 2}),
            "GWP visual RoPE positions changed");

    const semantics::PromptPlan right = semantics::prepare_prompt(
        {5, 6, 0}, {1, 1, 0}, 10,
        wam::internal::policy::SequenceSide::right);
    require(right.active_token_ids == std::vector<std::int32_t>({5, 6}) &&
                right.valid_tokens == 2 &&
                right.padded_attention_mask.size() == 64,
            "right-padded prompt plan changed");
    const semantics::PromptPlan left = semantics::prepare_prompt(
        {0, 5, 6}, {0, 1, 1}, 10,
        wam::internal::policy::SequenceSide::left);
    require(left.active_token_ids == std::vector<std::int32_t>({5, 6}),
            "left-padded prompt plan changed");
    require_error(
        [] {
            (void) semantics::prepare_prompt(
                {1, 2, 3}, {1, 0, 1}, 10,
                wam::internal::policy::SequenceSide::right);
        },
        wam::ErrorCode::invalid_argument, "noncontiguous prompt mask");

    const wam::internal::gwp05::FlowMatchEulerSchedule schedule =
        wam::internal::gwp05::make_flow_match_euler_schedule(10, 5.0F);
    require(schedule.steps() == 10 && schedule.sigmas.size() == 11 &&
                std::fabs(schedule.timesteps[6] - 717.31744F) < 1.0e-4F &&
                schedule.delta_sigma(0) < 0.0F && schedule.sigmas.back() == 0.0F,
            "flow-match schedule changed");
    std::vector<float> action = {1.0F, -2.0F};
    wam::internal::gwp05::flow_match_euler_step(
        action, {4.0F, 0.5F}, -0.25F);
    require(action == std::vector<float>({0.0F, -2.125F}),
            "flow-match Euler update changed");

    wam::test::TempFile file("gwp05-semantics-policy");
    wam::test::MetadataFixture fixture =
        wam::test::valid_gwp05_policy_fixture();
    fixture.write(file.string());
    const auto artifact =
        wam::internal::artifact::ArtifactView::open(file.string());
    auto spec = *wam::internal::policy::try_read_policy_spec(artifact);
    semantics::validate_policy_semantics(spec);

    spec.images.composition.width = 64;
    require_error([&] { semantics::validate_policy_semantics(spec); },
                  wam::ErrorCode::invalid_argument,
                  "GWP canvas gap must be rejected");
    return 0;
}
