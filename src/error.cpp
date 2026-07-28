#include "wam/error.h"

#include <utility>

namespace wam {

Error::Error(ErrorCode code, std::string message,
             std::vector<ErrorDetail> details)
    : std::runtime_error(std::move(message)),
      code_(code),
      details_(std::move(details)) {}

ErrorCode Error::code() const noexcept {
    return code_;
}

const std::vector<ErrorDetail> & Error::details() const noexcept {
    return details_;
}

} // namespace wam
