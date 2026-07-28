#pragma once

#include "wam/runtime_config.h"

#include <cstdarg>
#include <string>
#include <string_view>

namespace wam::internal::runtime {

class Logger final {
public:
    explicit Logger(const RuntimeConfig & config);

    bool enabled(LogLevel level) const noexcept;
    void log(LogLevel level, std::string_view message) const;
    void logf(LogLevel level, const char * format, ...) const;
    void logv(LogLevel level, const char * format,
              std::va_list arguments) const;

private:
    LogLevel level_ = LogLevel::warning;
    LoggerCallback callback_;
};

} // namespace wam::internal::runtime
