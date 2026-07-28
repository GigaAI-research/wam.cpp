#pragma once

#include "wam/error.h"

#include <iostream>
#include <string>

namespace wam::apps {

inline const char * error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::ok: return "ok";
        case ErrorCode::invalid_argument: return "invalid_argument";
        case ErrorCode::not_found: return "not_found";
        case ErrorCode::unsupported: return "unsupported";
        case ErrorCode::incompatible_artifact:
            return "incompatible_artifact";
        case ErrorCode::resource_exhausted: return "resource_exhausted";
        case ErrorCode::failed_precondition: return "failed_precondition";
        case ErrorCode::inference_failed: return "inference_failed";
        case ErrorCode::internal: return "internal";
    }
    return "unknown";
}

inline std::string json_string(const std::string & value) {
    std::string output = "\"";
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (character < 0x20U) {
                    constexpr char digits[] = "0123456789abcdef";
                    output += "\\u00";
                    output.push_back(digits[character >> 4U]);
                    output.push_back(digits[character & 0x0fU]);
                } else {
                    output.push_back(static_cast<char>(character));
                }
        }
    }
    output.push_back('"');
    return output;
}

inline int report_error(const Error & error) {
    std::cerr << "{\"valid\":false,\"error\":{\"code\":"
              << json_string(error_code_name(error.code()))
              << ",\"code_value\":"
              << static_cast<std::uint32_t>(error.code())
              << ",\"message\":" << json_string(error.what())
              << ",\"details\":[";
    for (std::size_t index = 0; index < error.details().size(); ++index) {
        if (index != 0) std::cerr << ',';
        std::cerr << "{\"field\":"
                  << json_string(error.details()[index].field)
                  << ",\"reason\":"
                  << json_string(error.details()[index].reason) << '}';
    }
    std::cerr << "]}}\n";
    return 1;
}

} // namespace wam::apps
