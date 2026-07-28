#include "cli_common.h"

#include "wam/model.h"

#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: wam-validate ARTIFACT_OR_BUNDLE\n";
        return 2;
    }
    try {
        wam::RuntimeConfig config;
        config.backend = wam::Backend::cpu_metadata;
        wam::Model model = wam::Model::load(argv[1], config);
        const wam::ModelInfo & info = model.info();
        std::cout << "{\"valid\":true,\"architecture\":"
                  << wam::apps::json_string(info.architecture)
                  << ",\"model\":"
                  << wam::apps::json_string(info.artifact_path)
                  << ",\"artifact_bytes\":" << info.artifact_bytes
                  << ",\"profile\":";
        if (info.policy_spec != nullptr) {
            std::cout << wam::apps::json_string(
                info.policy_spec->identity.profile);
        } else {
            std::cout << "null";
        }
        std::cout << "}\n";
        return 0;
    } catch (const wam::Error & error) {
        return wam::apps::report_error(error);
    }
}
