#pragma once

#include "policy/policy_spec.h"
#include "wam/types.h"

#include <memory>

namespace wam::internal {

class SessionImpl {
public:
    virtual ~SessionImpl();
    virtual Prediction predict(const Inputs & inputs) = 0;
    virtual Status reset() = 0;
};

class ModelImpl {
public:
    ModelImpl(ModelInfo info, policy::PolicySpecDraft policy_spec);
    virtual ~ModelImpl();

    const ModelInfo & info() const noexcept;
    const policy::PolicySpecDraft & policy_spec() const noexcept;
    virtual std::unique_ptr<SessionImpl> create_session(
        const SessionOptions & options) = 0;

protected:
    ModelInfo info_;
    policy::PolicySpecDraft policy_spec_;
};

} // namespace wam::internal
