#include "model_internal.h"

#include <utility>

namespace wam::internal {

SessionImpl::~SessionImpl() = default;

ModelImpl::ModelImpl(ModelInfo info, policy::PolicySpecDraft policy_spec)
    : info_(std::move(info)), policy_spec_(std::move(policy_spec)) {}

ModelImpl::~ModelImpl() = default;

const ModelInfo & ModelImpl::info() const noexcept {
    return info_;
}

const policy::PolicySpecDraft & ModelImpl::policy_spec() const noexcept {
    return policy_spec_;
}

} // namespace wam::internal
