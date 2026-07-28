#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace wam {

enum class ErrorCode : std::uint32_t {
    ok = 0,
    invalid_argument,
    not_found,
    unsupported,
    incompatible_artifact,
    resource_exhausted,
    failed_precondition,
    inference_failed,
    internal,
};

struct ErrorDetail {
    std::string field;
    std::string reason;
};

class Error final : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message,
          std::vector<ErrorDetail> details = {});

    ErrorCode code() const noexcept;
    const std::vector<ErrorDetail> & details() const noexcept;

private:
    ErrorCode code_;
    std::vector<ErrorDetail> details_;
};

} // namespace wam
