#include "runtime/logger.h"

#include <array>
#include <cstdio>
#include <vector>

namespace wam::internal::runtime {
namespace {

const char * level_name(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::error: return "error";
        case LogLevel::warning: return "warning";
        case LogLevel::info: return "info";
        case LogLevel::debug: return "debug";
    }
    return "unknown";
}

} // namespace

Logger::Logger(const RuntimeConfig & config)
    : level_(config.log_level), callback_(config.logger) {}

bool Logger::enabled(LogLevel level) const noexcept {
    return static_cast<std::uint32_t>(level) <=
        static_cast<std::uint32_t>(level_);
}

void Logger::log(LogLevel level, std::string_view message) const {
    if (!enabled(level)) return;
    if (callback_) {
        callback_(level, message);
        return;
    }
    std::fprintf(stderr, "wam[%s]: %.*s\n", level_name(level),
                 static_cast<int>(message.size()), message.data());
}

void Logger::logf(LogLevel level, const char * format, ...) const {
    if (!enabled(level)) return;
    std::va_list arguments;
    va_start(arguments, format);
    logv(level, format, arguments);
    va_end(arguments);
}

void Logger::logv(LogLevel level, const char * format,
                  std::va_list arguments) const {
    std::array<char, 512> stack{};
    std::va_list copy;
    va_copy(copy, arguments);
    const int required = std::vsnprintf(
        stack.data(), stack.size(), format, copy);
    va_end(copy);
    if (required < 0) {
        log(LogLevel::error, "cannot format log message");
        return;
    }
    if (static_cast<std::size_t>(required) < stack.size()) {
        log(level, std::string_view(stack.data(),
                                    static_cast<std::size_t>(required)));
        return;
    }
    std::vector<char> heap(static_cast<std::size_t>(required) + 1);
    std::vsnprintf(heap.data(), heap.size(), format, arguments);
    log(level, std::string_view(heap.data(), static_cast<std::size_t>(required)));
}

} // namespace wam::internal::runtime
