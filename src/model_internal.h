#pragma once

#include "policy/policy_spec.h"
#include "wam/model.h"
#include "wam/session.h"

#include <memory>

namespace wam::internal {

class SessionImpl {
public:
    virtual ~SessionImpl();
    virtual Prediction predict(const Observation & observation) = 0;
    virtual void reset() = 0;
};

class ModelImpl {
public:
    ModelImpl(ModelInfo info, PolicySpec policy_spec);
    virtual ~ModelImpl();

    const ModelInfo & info() const noexcept;
    const PolicySpec & policy_spec() const noexcept;
    virtual std::unique_ptr<SessionImpl> create_session(
        const SessionConfig & config) = 0;

protected:
    ModelInfo info_;
    PolicySpec policy_spec_;
};

struct ModelAccess {
    static Model adopt(std::unique_ptr<ModelImpl> impl);
    static const ModelImpl & impl(const Model & model);
};

Model adopt_model(std::unique_ptr<ModelImpl> impl);
const ModelImpl & model_impl(const Model & model);

} // namespace wam::internal
