#include "replay.h"

#include "json.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_bytes(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path);
    const std::streamsize size = input.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("action file is not F32");
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(result.data()), size)) {
        throw std::runtime_error("cannot read " + path);
    }
    return result;
}

void usage(const char * program) {
    std::cerr << "usage: " << program << " --actual ACTION.f32 --expected ACTION.f32\n"
              << "       [--mean-tolerance 0.001] [--max-tolerance 0.01] [--report FILE|-]\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::string actual_path, expected_path, report_path = "-";
        double mean_tolerance = 1.0e-3, max_tolerance = 1.0e-2;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto value = [&]() -> std::string {
                if (++index >= argc) throw std::runtime_error(argument + " requires a value");
                return argv[index];
            };
            if (argument == "--actual") actual_path = value();
            else if (argument == "--expected") expected_path = value();
            else if (argument == "--mean-tolerance") mean_tolerance = std::stod(value());
            else if (argument == "--max-tolerance") max_tolerance = std::stod(value());
            else if (argument == "--report") report_path = value();
            else if (argument == "--help" || argument == "-h") { usage(argv[0]); return 0; }
            else throw std::runtime_error("unknown argument: " + argument);
        }
        if (actual_path.empty() || expected_path.empty() || mean_tolerance < 0.0 ||
            max_tolerance < 0.0) {
            usage(argv[0]);
            return 2;
        }
        wam::Tensor actual;
        actual.data = read_bytes(actual_path);
        actual.dtype = wam::DType::f32;
        actual.byte_order = wam::ByteOrder::little;
        actual.layout = "flat";
        actual.shape.push_back(
            static_cast<std::int64_t>(actual.data.size() / sizeof(float)));
        const wam::apps::Comparison comparison =
            wam::apps::compare_action(actual, expected_path);
        const bool pass = comparison.finite && comparison.mean_abs <= mean_tolerance &&
            comparison.max_abs <= max_tolerance;
        nlohmann::json report = {
            {"format", "wam-action-comparison-v1"}, {"actual", actual_path},
            {"expected", expected_path},
            {"actual_sha256", wam::apps::sha256_hex(actual.data.data(), actual.data.size())},
            {"comparison", wam::apps::comparison_json(comparison)},
            {"gates", {{"mean_abs", mean_tolerance}, {"max_abs", max_tolerance}}},
            {"pass", pass}
        };
        wam::apps::write_json(report, report_path);
        return pass ? 0 : 4;
    } catch (const std::exception & error) {
        std::cerr << "wam-compare: " << error.what() << '\n';
        return 1;
    }
}
