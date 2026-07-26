#include "models/fastwam/artifact.h"
#include "models/fastwam/semantics.h"
#include "support/test_utils.h"

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
