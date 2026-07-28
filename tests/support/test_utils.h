#pragma once

#include "wam/wam.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace wam::test {

inline void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_error(Function && function, ErrorCode expected,
                   const std::string & context) {
    try {
        std::forward<Function>(function)();
    } catch (const Error & error) {
        require(error.code() == expected,
                context + ": unexpected wam::Error code");
        return;
    } catch (...) {
        throw std::runtime_error(context + ": unexpected exception type");
    }
    throw std::runtime_error(context + ": expected wam::Error");
}

} // namespace wam::test
