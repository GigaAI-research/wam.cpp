#include "model_internal.h"

#include <utility>

namespace wam::internal {

SessionImpl::~SessionImpl() = default;

ModelImpl::ModelImpl(ModelInfo info, PolicySpec policy_spec)
    : info_(std::move(info)),
      policy_spec_(std::move(policy_spec)) {
    info_.policy_spec = std::make_shared<const PolicySpec>(policy_spec_);
}

ModelImpl::~ModelImpl() = default;

const ModelInfo & ModelImpl::info() const noexcept {
    return info_;
}

const PolicySpec & ModelImpl::policy_spec() const noexcept {
    return policy_spec_;
}

} // namespace wam::internal
