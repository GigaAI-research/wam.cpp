#pragma once

#include "models/fastwam/artifact.h"
#include "wam/types.h"

#include <random>

namespace wam::internal::fastwam {

struct PreparedInputs;

PreparedInputs prepare_inputs(const Inputs & inputs,
                              const ArtifactContract & artifact,
                              std::mt19937 & random);

} // namespace wam::internal::fastwam
