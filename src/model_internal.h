#pragma once

#include "wam/types.h"

#include <memory>
#include <utility>

namespace wam::internal {

class SessionImpl {
public:
    virtual ~SessionImpl() = default;
    virtual Prediction predict(const Inputs & inputs) = 0;
    virtual Status reset() = 0;
};

class ModelImpl {
public:
    explicit ModelImpl(ModelInfo info) : info_(std::move(info)) {}
    virtual ~ModelImpl() = default;

    const ModelInfo & info() const noexcept { return info_; }
    virtual std::unique_ptr<SessionImpl> create_session(
        const SessionOptions & options) = 0;

protected:
    ModelInfo info_;
};

} // namespace wam::internal
