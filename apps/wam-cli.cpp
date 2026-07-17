#include "replay.h"

#include "json.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace {

void usage(const char * program) {
    std::cerr << "usage: " << program << " --model MODEL.gguf --request REQUEST.pb|json\n"
              << "       --precision f32|bf16 --action FILE [--action-format f32|json]\n"
              << "       [--report FILE|-] [--expected EXPECTED.f32] [--device INDEX]\n"
              << "       [--disable-prefix-cache] [--prompt-cache-capacity N]\n"
              << "       [--language-policy resident|fixed|external_embedding]\n";
}

struct ModelDeleter { void operator()(wam::Model * value) const { wam::model_free(value); } };
struct SessionDeleter { void operator()(wam::Session * value) const { wam::session_free(value); } };

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path, request_path, action_path, action_format, report_path = "-";
        std::string expected_path, precision_text;
        std::string language_policy_text = "resident";
        std::int32_t device = 0;
        std::size_t prompt_cache_capacity = 4;
        bool prefix_cache = true;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto value = [&]() -> std::string {
                if (++index >= argc) throw std::runtime_error(argument + " requires a value");
                return argv[index];
            };
            if (argument == "--model") model_path = value();
            else if (argument == "--request") request_path = value();
            else if (argument == "--precision") precision_text = value();
            else if (argument == "--action") action_path = value();
            else if (argument == "--action-format") action_format = value();
            else if (argument == "--report") report_path = value();
            else if (argument == "--expected") expected_path = value();
            else if (argument == "--device") device = std::stoi(value());
            else if (argument == "--language-policy") language_policy_text = value();
            else if (argument == "--prompt-cache-capacity") {
                prompt_cache_capacity = static_cast<std::size_t>(std::stoull(value()));
            } else if (argument == "--disable-prefix-cache") prefix_cache = false;
            else if (argument == "--help" || argument == "-h") { usage(argv[0]); return 0; }
            else throw std::runtime_error("unknown argument: " + argument);
        }
        if (model_path.empty() || request_path.empty() || action_path.empty() ||
            precision_text.empty()) {
            usage(argv[0]);
            return 2;
        }
        if (action_format.empty()) {
            action_format = std::filesystem::path(action_path).extension() == ".json"
                ? "json" : "f32";
        }

        const wam::Precision precision = wam::apps::parse_precision(precision_text);
        const wam::apps::ReplayRequest request = wam::apps::load_request(request_path);
        wam::ModelOptions options;
        options.artifact_path = model_path;
        options.backend = wam::Backend::cuda;
        options.precision = precision;
        options.device_index = device;
        options.prompt_cache_capacity = prompt_cache_capacity;
        options.language_encoder_policy =
            wam::apps::parse_language_policy(language_policy_text);
        if (options.language_encoder_policy == wam::LanguageEncoderPolicy::fixed) {
            if (request.token_ids.empty()) {
                throw std::runtime_error("fixed policy requires a token-input request");
            }
            options.fixed_prompt = wam::FixedPrompt{request.token_ids, request.attention_mask};
        }

        const auto load_begin = std::chrono::steady_clock::now();
        std::unique_ptr<wam::Model, ModelDeleter> model(wam::model_load(options));
        const double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - load_begin).count();
        std::unique_ptr<wam::Session, SessionDeleter> session(
            wam::session_create(model.get(), {prefix_cache}));
        const wam::Prediction prediction = wam::predict(session.get(), request.inputs());
        wam::apps::write_action(prediction.action, action_path, action_format);
        const wam::apps::Comparison comparison = expected_path.empty()
            ? wam::apps::Comparison{}
            : wam::apps::compare_action(prediction.action, expected_path);

        const auto runtime = wam::runtime_version();
        const auto protocol = wam::protocol_version();
        nlohmann::json report = {
            {"format", "wam-cli-v1"},
            {"software", {{"runtime_version", std::to_string(runtime.major) + "." +
                                                std::to_string(runtime.minor) + "." +
                                                std::to_string(runtime.patch)},
                           {"protocol_version", std::to_string(protocol.major) + "." +
                                                 std::to_string(protocol.minor) + "." +
                                                 std::to_string(protocol.patch)},
                           {"artifact_schema_version", wam::artifact_schema_version()},
                           {"abi_version", wam::abi_version()}}},
            {"device", wam::apps::device_json(device)},
            {"model", wam::apps::model_json(wam::model_info(model.get()))},
            {"request", {{"path", request_path}, {"format", request.format}}},
            {"load_ms", load_ms},
            {"action", wam::apps::action_json(prediction.action, false)},
            {"action_path", action_path},
            {"stats", wam::apps::stats_json(prediction.stats)},
            {"comparison", wam::apps::comparison_json(comparison)}
        };
        wam::apps::write_json(report, report_path);
        if (!wam::apps::action_json(prediction.action, false).at("finite").get<bool>()) return 3;
        if (comparison.present) {
            const double max_gate = precision == wam::Precision::bf16_latency ? 1.0e-2 : 1.0e-3;
            if (!comparison.finite || comparison.mean_abs > 1.0e-3 ||
                comparison.max_abs > max_gate) return 4;
        }
        return 0;
    } catch (const wam::Error & error) {
        std::cerr << "wam-cli: error code=" << static_cast<std::uint32_t>(error.code())
                  << " message=" << error.what() << '\n';
    } catch (const std::exception & error) {
        std::cerr << "wam-cli: " << error.what() << '\n';
    }
    return 1;
}
