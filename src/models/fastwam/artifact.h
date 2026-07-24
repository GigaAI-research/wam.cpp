#pragma once

#include "policy/policy_spec.h"

#include <memory>

namespace wam::internal {

class GgufReader;

namespace fastwam {

struct ArtifactContract;

std::shared_ptr<ArtifactContract> load_artifact(
    std::shared_ptr<GgufReader> reader,
    const policy::PolicySpecDraft & policy_spec);
void validate_artifact(const ArtifactContract & artifact,
                       const policy::PolicySpecDraft & policy_spec);

} // namespace fastwam
} // namespace wam::internal
