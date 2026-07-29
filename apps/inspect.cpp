#include "cli_common.h"

#include "artifact/artifact_view.h"
#include "policy/policy_spec.h"

#include <iostream>
#include <optional>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: wam-inspect ARTIFACT_OR_BUNDLE\n";
        return 2;
    }
    try {
        const wam::internal::artifact::ArtifactView artifact =
            wam::internal::artifact::ArtifactView::open(argv[1]);
        const std::optional<wam::PolicySpec> policy_spec =
            wam::internal::policy::try_read_policy_spec(artifact);
        const auto & bundle = artifact.bundle();
        std::cout << "{\"valid\":true,\"input\":"
                  << wam::apps::json_string(bundle.input_path.string())
                  << ",\"model\":"
                  << wam::apps::json_string(artifact.path())
                  << ",\"bundle\":"
                  << (bundle.uses_manifest ? "true" : "false")
                  << ",\"architecture\":"
                  << wam::apps::json_string(
                         artifact.require_string("general.architecture"))
                  << ",\"artifact_bytes\":" << artifact.file_size()
                  << ",\"tensor_count\":" << artifact.tensors().size()
                  << ",\"resources\":{\"tokenizer\":";
        if (bundle.tokenizer_path.has_value()) {
            std::cout << wam::apps::json_string(
                bundle.tokenizer_path->string());
        } else {
            std::cout << "null";
        }
        std::cout << ",\"language_encoder\":";
        if (bundle.language_encoder_path.has_value()) {
            std::cout << wam::apps::json_string(
                bundle.language_encoder_path->string());
        } else {
            std::cout << "null";
        }
        std::cout << '}'
                  << ",\"policy_spec\":";
        if (policy_spec.has_value()) {
            std::vector<std::string> image_roles;
            image_roles.reserve(policy_spec->images.views.size());
            for (const wam::ImageTransformSpec & view :
                 policy_spec->images.views) {
                image_roles.push_back(view.role);
            }
            std::cout << "{\"schema_version\":"
                      << policy_spec->identity.artifact_schema_version
                      << ",\"profile\":"
                      << wam::apps::json_string(policy_spec->identity.profile)
                      << ",\"image_roles\":";
            wam::apps::write_json_strings(std::cout, image_roles);
            std::cout << ",\"state_dim\":" << policy_spec->state.real_dim
                      << ",\"state_fields\":";
            wam::apps::write_json_strings(
                std::cout, policy_spec->state.fields);
            std::cout << ",\"language_max_tokens\":"
                      << policy_spec->language.max_tokens
                      << ",\"action_horizon\":"
                      << policy_spec->action.horizon
                      << ",\"action_dim\":"
                      << policy_spec->action.real_dim
                      << ",\"action_fields\":";
            wam::apps::write_json_strings(
                std::cout, policy_spec->action.fields);
            std::cout << '}';
        } else {
            std::cout << "null";
        }
        std::cout << "}\n";
        return 0;
    } catch (const wam::Error & error) {
        return wam::apps::report_error(error);
    }
}
