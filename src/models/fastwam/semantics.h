#pragma once

#include "models/fastwam/artifact.h"

namespace wam::internal::fastwam {

struct SequenceGeometry;

void validate_policy_semantics(const policy::PolicySpecDraft & policy_spec);
SequenceGeometry resolve_sequence_geometry(const ArtifactContract & artifact,
                                           const policy::PolicySpecDraft & policy_spec);

} // namespace wam::internal::fastwam
