#include "artifact/tensor_spec.h"

#include "wam/error.h"

#include <algorithm>
#include <sstream>

namespace wam::internal::artifact {
namespace {

std::vector<std::int64_t> normalized(std::vector<std::int64_t> shape) {
    while (shape.size() > 1 && shape.back() == 1) shape.pop_back();
    return shape;
}

std::string shape_string(const std::vector<std::int64_t> & shape) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) output << ',';
        output << shape[index];
    }
    output << ']';
    return output.str();
}

} // namespace

template <typename Artifact>
void validate_tensor_impl(const Artifact & artifact, const TensorSpec & spec) {
    if (spec.name.empty()) {
        throw Error(ErrorCode::invalid_argument,
                    "TensorSpec name must not be empty",
                    {{"tensor_spec.name", "empty"}});
    }
    const GgufTensorInfo * tensor = artifact.find_tensor(spec.name);
    if (tensor == nullptr) {
        if (!spec.required) return;
        throw Error(ErrorCode::incompatible_artifact,
                    "required artifact tensor is missing",
                    {{spec.name, "missing"}});
    }
    if (!spec.accepted_dtypes.empty() &&
        std::find(spec.accepted_dtypes.begin(), spec.accepted_dtypes.end(),
                  tensor->dtype) == spec.accepted_dtypes.end()) {
        throw Error(ErrorCode::incompatible_artifact,
                    "artifact tensor has an unsupported dtype",
                    {{spec.name, "dtype mismatch"}});
    }
    if (spec.shape.has_value() &&
        normalized(tensor->shape) != normalized(*spec.shape)) {
        throw Error(ErrorCode::incompatible_artifact,
                    "artifact tensor has the wrong shape",
                    {{spec.name,
                      "expected " + shape_string(*spec.shape) + ", got " +
                          shape_string(tensor->shape)}});
    }
}

void validate_tensor(const ArtifactView & artifact, const TensorSpec & spec) {
    validate_tensor_impl(artifact, spec);
}

void validate_tensor(const GgufReader & artifact, const TensorSpec & spec) {
    validate_tensor_impl(artifact, spec);
}

void validate_tensors(const ArtifactView & artifact,
                      const std::vector<TensorSpec> & specs) {
    for (const TensorSpec & spec : specs) validate_tensor(artifact, spec);
}

void validate_tensors(const GgufReader & artifact,
                      const std::vector<TensorSpec> & specs) {
    for (const TensorSpec & spec : specs) validate_tensor(artifact, spec);
}

} // namespace wam::internal::artifact
