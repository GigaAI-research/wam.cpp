#pragma once

#include "artifact/artifact_view.h"

#include <optional>
#include <string>
#include <vector>

namespace wam::internal::artifact {

struct TensorSpec {
    std::string name;
    std::vector<DType> accepted_dtypes;
    std::optional<std::vector<std::int64_t>> shape;
    bool required = true;
};

void validate_tensor(const ArtifactView & artifact, const TensorSpec & spec);
void validate_tensor(const GgufReader & artifact, const TensorSpec & spec);
void validate_tensors(const ArtifactView & artifact,
                      const std::vector<TensorSpec> & specs);
void validate_tensors(const GgufReader & artifact,
                      const std::vector<TensorSpec> & specs);

} // namespace wam::internal::artifact
