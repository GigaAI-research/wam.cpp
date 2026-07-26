#pragma once

#include "engine_internal.h"
#include "models/fastwam/artifact.h"

#include "ggml.h"

#include <vector>

namespace wam::internal::fastwam {

std::vector<ggml_bf16_t> encode_first_frame(
    Engine & engine, const ArtifactContract & artifact,
    const std::vector<float> & patchified_pixels);

} // namespace wam::internal::fastwam
