#include "replay.h"

#include "json.hpp"

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void usage(const char * program) {
    std::cerr << "usage: " << program << " --model MODEL.gguf --request REQUEST.pb|json\n"
              << "       --precision f32|bf16 --mode cold|raw|steady --report FILE|-\n"
              << "       [--expected EXPECTED.f32] [--action FILE] [--warmup 3]\n"
              << "       [--repetitions 15] [--device INDEX] [--disable-prefix-cache]\n"
              << "       [--language-policy resident|fixed|external_embedding]\n";
}

struct ModelDeleter { void operator()(wam::Model * value) const { wam::model_free(value); } };
struct SessionDeleter { void operator()(wam::Session * value) const { wam::session_free(value); } };

void add_stats(std::map<std::string, std::vector<double>> & values,
               const wam::Stats & stats) {
    values["preprocess"].push_back(stats.preprocess_milliseconds);
    values["vae"].push_back(stats.vae_milliseconds);
    values["text_encoder"].push_back(stats.text_encoder_milliseconds);
    values["prompt_projection"].push_back(stats.prompt_projection_milliseconds);
    values["prefix"].push_back(stats.prefix_milliseconds);
    values["denoise"].push_back(stats.denoise_milliseconds);
    values["postprocess"].push_back(stats.postprocess_milliseconds);
    values["total"].push_back(stats.total_milliseconds);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string model_path, request_path, precision_text, mode, report_path;
        std::string expected_path, action_path;
        std::string language_policy_text = "resident";
        int warmup = 3, repetitions = 15;
        std::int32_t device = 0;
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
            else if (argument == "--mode") mode = value();
            else if (argument == "--report") report_path = value();
            else if (argument == "--expected") expected_path = value();
            else if (argument == "--action") action_path = value();
            else if (argument == "--warmup") warmup = std::stoi(value());
            else if (argument == "--repetitions") repetitions = std::stoi(value());
            else if (argument == "--device") device = std::stoi(value());
            else if (argument == "--language-policy") language_policy_text = value();
            else if (argument == "--disable-prefix-cache") prefix_cache = false;
            else if (argument == "--help" || argument == "-h") { usage(argv[0]); return 0; }
            else throw std::runtime_error("unknown argument: " + argument);
        }
        if (model_path.empty() || request_path.empty() || precision_text.empty() ||
            report_path.empty() || (mode != "cold" && mode != "raw" && mode != "steady") ||
            warmup < 0 || repetitions < 1) {
            usage(argv[0]);
            return 2;
        }
        if (mode == "cold") { warmup = 0; repetitions = 1; }

        const wam::Precision precision = wam::apps::parse_precision(precision_text);
        const wam::apps::ReplayRequest request = wam::apps::load_request(request_path);
        wam::ModelOptions options;
        options.artifact_path = model_path;
        options.backend = wam::Backend::cuda;
        options.precision = precision;
        options.device_index = device;
        options.prompt_cache_capacity = mode == "steady" ? 4 : 0;
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

        wam::Prediction prediction;
        wam::Stats cold_stats;
        std::map<std::string, std::vector<double>> timings;
        std::vector<std::vector<double>> denoise_steps;
        int prompt_hits = 0, projected_hits = 0;
        for (int run = -1; run < warmup + repetitions; ++run) {
            prediction = wam::predict(session.get(), request.inputs());
            if (run == -1) {
                cold_stats = prediction.stats;
                if (mode == "cold") break;
                continue;
            }
            if (run < warmup) continue;
            add_stats(timings, prediction.stats);
            prompt_hits += prediction.stats.prompt_cache_hit ? 1 : 0;
            projected_hits += prediction.stats.projected_prompt_cache_hit ? 1 : 0;
            if (denoise_steps.empty()) {
                denoise_steps.resize(prediction.stats.denoise_step_milliseconds.size());
            }
            for (std::size_t step = 0;
                 step < denoise_steps.size() &&
                 step < prediction.stats.denoise_step_milliseconds.size(); ++step) {
                denoise_steps[step].push_back(prediction.stats.denoise_step_milliseconds[step]);
            }
        }
        if (mode == "cold") {
            add_stats(timings, prediction.stats);
            denoise_steps.resize(prediction.stats.denoise_step_milliseconds.size());
            for (std::size_t step = 0; step < denoise_steps.size(); ++step) {
                denoise_steps[step].push_back(prediction.stats.denoise_step_milliseconds[step]);
            }
        }
        if (!action_path.empty()) wam::apps::write_action(prediction.action, action_path, "f32");
        const wam::apps::Comparison comparison = expected_path.empty()
            ? wam::apps::Comparison{}
            : wam::apps::compare_action(prediction.action, expected_path);

        nlohmann::json timing_json = nlohmann::json::object();
        for (const auto & item : timings) {
            timing_json[item.first] = wam::apps::sample_summary(item.second);
        }
        nlohmann::json step_json = nlohmann::json::array();
        for (const auto & samples : denoise_steps) {
            step_json.push_back(wam::apps::sample_summary(samples));
        }
        const auto runtime = wam::runtime_version();
        const auto protocol = wam::protocol_version();
        nlohmann::json report = {
            {"format", "wam-benchmark-v1"}, {"mode", mode},
            {"protocol", {{"warmup", warmup}, {"repetitions", repetitions},
                           {"batch", 1}, {"denoise_steps", 10}}},
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
            {"load_ms", load_ms}, {"peak_rss_kib", wam::apps::peak_rss_kib()},
            {"cold_request", wam::apps::stats_json(cold_stats)},
            {"timing_ms", timing_json}, {"denoise_steps_ms", step_json},
            {"cache", {{"prompt_hits", prompt_hits},
                        {"projected_prompt_hits", projected_hits}}},
            {"action", wam::apps::action_json(prediction.action, false)},
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
        std::cerr << "wam-bench: error code=" << static_cast<std::uint32_t>(error.code())
                  << " message=" << error.what() << '\n';
    } catch (const std::exception & error) {
        std::cerr << "wam-bench: " << error.what() << '\n';
    }
    return 1;
}
