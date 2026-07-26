#pragma once

#include "engine_internal.h"
#include "models/fastwam/artifact.h"
#include "models/fastwam/inputs.h"

#include "wam/types.h"

namespace wam::internal::fastwam {

CoreAction run_pipeline(Engine & engine, const ArtifactContract & artifact,
                        const PreparedInputs & inputs);

} // namespace wam::internal::fastwam
