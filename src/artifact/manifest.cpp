#include "artifact/manifest.h"

#include "wam/error.h"

#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

namespace wam::internal::artifact {
namespace {

using JsonValue = std::variant<std::string, std::uint64_t, std::nullptr_t>;

[[noreturn]] void manifest_error(const std::string & message,
                                 const std::string & field,
                                 const std::string & reason) {
    throw Error(ErrorCode::incompatible_artifact, message,
                {{field, reason}});
}

class JsonObjectParser final {
public:
    explicit JsonObjectParser(std::string_view input) : input_(input) {}

    std::map<std::string, JsonValue> parse() {
        std::map<std::string, JsonValue> result;
        whitespace();
        expect('{');
        whitespace();
        if (consume('}')) return result;
        for (;;) {
            whitespace();
            const std::string key = string();
            if (!result.emplace(key, nullptr).second) {
                fail("duplicate key " + key);
            }
            whitespace();
            expect(':');
            whitespace();
            result[key] = value();
            whitespace();
            if (consume('}')) break;
            expect(',');
        }
        whitespace();
        if (offset_ != input_.size()) fail("trailing content");
        return result;
    }

private:
    [[noreturn]] void fail(const std::string & reason) const {
        manifest_error("bundle manifest is not valid JSON", "manifest.json",
                       reason + " at byte " + std::to_string(offset_));
    }

    void whitespace() {
        while (offset_ < input_.size() &&
               std::isspace(static_cast<unsigned char>(input_[offset_]))) {
            ++offset_;
        }
    }

    bool consume(char expected) {
        if (offset_ < input_.size() && input_[offset_] == expected) {
            ++offset_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    static void append_utf8(std::string & output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(
                static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        }
    }

    std::uint32_t hex4() {
        if (input_.size() - offset_ < 4) fail("truncated unicode escape");
        std::uint32_t value = 0;
        for (int index = 0; index < 4; ++index) {
            const char character = input_[offset_++];
            value <<= 4U;
            if (character >= '0' && character <= '9') {
                value += static_cast<std::uint32_t>(character - '0');
            } else if (character >= 'a' && character <= 'f') {
                value += static_cast<std::uint32_t>(character - 'a' + 10);
            } else if (character >= 'A' && character <= 'F') {
                value += static_cast<std::uint32_t>(character - 'A' + 10);
            } else {
                fail("invalid unicode escape");
            }
        }
        return value;
    }

    std::string string() {
        expect('"');
        std::string output;
        while (offset_ < input_.size()) {
            const unsigned char character =
                static_cast<unsigned char>(input_[offset_++]);
            if (character == '"') return output;
            if (character < 0x20U) fail("control character in string");
            if (character != '\\') {
                output.push_back(static_cast<char>(character));
                continue;
            }
            if (offset_ == input_.size()) fail("truncated escape");
            switch (input_[offset_++]) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    const std::uint32_t codepoint = hex4();
                    if (codepoint >= 0xd800U && codepoint <= 0xdfffU) {
                        fail("unicode surrogate is unsupported");
                    }
                    append_utf8(output, codepoint);
                    break;
                }
                default: fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    JsonValue value() {
        if (offset_ == input_.size()) fail("missing value");
        if (input_[offset_] == '"') return string();
        if (input_.substr(offset_, 4) == "null") {
            offset_ += 4;
            return nullptr;
        }
        if (!std::isdigit(static_cast<unsigned char>(input_[offset_]))) {
            fail("only string, unsigned integer, and null values are allowed");
        }
        std::uint64_t result = 0;
        do {
            const std::uint64_t digit =
                static_cast<std::uint64_t>(input_[offset_] - '0');
            if (result > (std::numeric_limits<std::uint64_t>::max() - digit) /
                             10U) {
                fail("integer overflow");
            }
            result = result * 10U + digit;
            ++offset_;
        } while (offset_ < input_.size() &&
                 std::isdigit(static_cast<unsigned char>(input_[offset_])));
        return result;
    }

    std::string_view input_;
    std::size_t offset_ = 0;
};

std::string required_string(const std::map<std::string, JsonValue> & object,
                            const std::string & field) {
    const auto item = object.find(field);
    if (item == object.end()) {
        manifest_error("bundle manifest field is required", field, "missing");
    }
    const auto * value = std::get_if<std::string>(&item->second);
    if (value == nullptr || value->empty()) {
        manifest_error("bundle manifest field must be a nonempty string", field,
                       "wrong type or empty");
    }
    return *value;
}

std::optional<std::string> optional_string(
    const std::map<std::string, JsonValue> & object,
    const std::string & field) {
    const auto item = object.find(field);
    if (item == object.end() ||
        std::holds_alternative<std::nullptr_t>(item->second)) {
        return std::nullopt;
    }
    const auto * value = std::get_if<std::string>(&item->second);
    if (value == nullptr || value->empty()) {
        manifest_error("bundle manifest resource must be a nonempty string or null",
                       field, "wrong type or empty");
    }
    return *value;
}

std::filesystem::path resolve_resource(const std::filesystem::path & root,
                                       const std::string & value,
                                       const std::string & field) {
    const std::filesystem::path relative(value);
    if (relative.empty() || relative.is_absolute() ||
        relative.has_root_name() || relative.has_root_directory()) {
        manifest_error("bundle resource path must be relative", field, value);
    }
    for (const auto & component : relative) {
        if (component == "..") {
            manifest_error("bundle resource cannot escape its root", field,
                           value);
        }
    }

    std::error_code error;
    const std::filesystem::path resolved =
        std::filesystem::weakly_canonical(root / relative, error);
    if (error || !std::filesystem::is_regular_file(resolved, error)) {
        throw Error(ErrorCode::not_found, "bundle resource is missing",
                    {{field, value}});
    }
    const std::filesystem::path canonical_root =
        std::filesystem::weakly_canonical(root, error);
    auto root_part = canonical_root.begin();
    auto path_part = resolved.begin();
    for (; root_part != canonical_root.end() && path_part != resolved.end();
         ++root_part, ++path_part) {
        if (*root_part != *path_part) {
            manifest_error("bundle resource resolves outside its root", field,
                           value);
        }
    }
    if (root_part != canonical_root.end()) {
        manifest_error("bundle resource resolves outside its root", field,
                       value);
    }
    return resolved;
}

} // namespace

ArtifactBundle resolve_bundle(const std::string & input_path) {
    if (input_path.empty()) {
        throw Error(ErrorCode::invalid_argument,
                    "artifact input path must not be empty",
                    {{"artifact_path", "empty"}});
    }

    std::error_code error;
    const std::filesystem::path input =
        std::filesystem::absolute(input_path, error);
    if (error) {
        throw Error(ErrorCode::not_found, "cannot resolve artifact input path",
                    {{"artifact_path", input_path}});
    }
    if (std::filesystem::is_regular_file(input, error)) {
        return {input, input.parent_path(), input, std::nullopt, std::nullopt,
                false};
    }
    if (!std::filesystem::is_directory(input, error)) {
        throw Error(ErrorCode::not_found,
                    "artifact input is neither a GGUF file nor a bundle directory",
                    {{"artifact_path", input_path}});
    }

    const std::filesystem::path manifest_path = input / "manifest.json";
    std::ifstream stream(manifest_path, std::ios::binary);
    if (!stream) {
        throw Error(ErrorCode::not_found, "bundle manifest is missing",
                    {{"manifest", manifest_path.string()}});
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        throw Error(ErrorCode::incompatible_artifact,
                    "cannot read bundle manifest",
                    {{"manifest", manifest_path.string()}});
    }
    const std::map<std::string, JsonValue> object =
        JsonObjectParser(contents.str()).parse();
    for (const auto & item : object) {
        if (item.first != "format" && item.first != "manifest_schema_version" &&
            item.first != "model" && item.first != "tokenizer" &&
            item.first != "language_encoder") {
            manifest_error("bundle manifest contains an unknown field",
                           item.first, "unsupported");
        }
    }
    if (required_string(object, "format") != "wam-bundle-v1") {
        manifest_error("unsupported bundle manifest format", "format",
                       "expected wam-bundle-v1");
    }
    const auto schema_item = object.find("manifest_schema_version");
    const auto * schema = schema_item == object.end()
        ? nullptr
        : std::get_if<std::uint64_t>(&schema_item->second);
    if (schema == nullptr || *schema != kBundleManifestSchemaVersion) {
        manifest_error("unsupported bundle manifest schema",
                       "manifest_schema_version", "expected 1");
    }

    ArtifactBundle bundle;
    bundle.input_path = input;
    bundle.root = std::filesystem::weakly_canonical(input, error);
    bundle.model_path = resolve_resource(
        bundle.root, required_string(object, "model"), "model");
    if (const auto tokenizer = optional_string(object, "tokenizer")) {
        bundle.tokenizer_path =
            resolve_resource(bundle.root, *tokenizer, "tokenizer");
    }
    if (const auto encoder = optional_string(object, "language_encoder")) {
        bundle.language_encoder_path =
            resolve_resource(bundle.root, *encoder, "language_encoder");
    }
    bundle.uses_manifest = true;
    return bundle;
}

} // namespace wam::internal::artifact
