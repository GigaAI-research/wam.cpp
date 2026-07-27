#include "models/fastwam/artifact.h"
#include "models/fastwam/engine/scheduler.h"
#include "models/fastwam/semantics.h"
#include "support/test_utils.h"

#include <vector>

using wam::internal::fastwam::ArtifactContract;
using wam::internal::fastwam::semantics::GeometryInput;
using wam::internal::fastwam::semantics::resolve_geometry;
using wam::test::require_error;
using wam::test::require;

int main() {
    const auto geometry = resolve_geometry(
        GeometryInput{224, 448, 16, 32, 30, 24, 128, 3072});
    require(geometry.latent_height == 14 && geometry.latent_width == 28,
            "FastWAM latent geometry is wrong");
    require(geometry.video_grid_height == 7 &&
                geometry.video_grid_width == 14 &&
                geometry.video_tokens == 98,
            "FastWAM visual token geometry is wrong");
    require(geometry.action_tokens == 32,
            "FastWAM action token geometry is wrong");

    const wam::internal::fastwam::FlowSchedule schedule =
        wam::internal::fastwam::make_inference_schedule(10, 5.0F);
    require(schedule.timesteps == std::vector<float>({
                1000.0F, 980.0F, 952.0F, 920.0F, 884.0F,
                832.0F, 768.0F, 680.0F, 556.0F, 358.0F}),
            "FastWAM BF16 donor timesteps changed");
    require(schedule.deltas == std::vector<float>({
                -0.021728515625F, -0.02587890625F, -0.03125F,
                -0.038818359375F, -0.049072265625F, -0.06396484375F,
                -0.08740234375F, -0.1259765625F, -0.1982421875F,
                -0.357421875F}),
            "FastWAM BF16 donor deltas changed");

    require_error(
        [] { (void) resolve_geometry(
            GeometryInput{224, 447, 16, 32, 30, 24, 128, 3072}); },
        wam::ErrorCode::incompatible_artifact,
        "fastwam.image_height");
    require_error(
        [] { (void) resolve_geometry(
            GeometryInput{224, 448, 16, 32, 30, 24, 64, 3072}); },
        wam::ErrorCode::incompatible_artifact,
        "fastwam.video_hidden_dim");
    return 0;
}
