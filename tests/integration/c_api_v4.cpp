#include "fixtures/metadata_fixture.h"
#include "support/temp_file.h"
#include "support/test_utils.h"

#include "wam/c_api.h"

#include <cstring>

int main() {
    using wam::test::require;

    require(wam_c_abi_version() == WAM_C_ABI_VERSION,
            "C ABI runtime version differs from the header");
    wam_c_model_options options;
    wam_c_model_options_init(&options);
    require(options.struct_version == WAM_C_STRUCT_VERSION &&
                options.struct_size == sizeof(options),
            "model option initializer did not publish its layout");

    wam_c_model * model = nullptr;
    wam_c_error * error = nullptr;
    options.struct_version = WAM_C_STRUCT_VERSION + 1;
    require(wam_c_model_create(&options, &model, &error) ==
                WAM_C_STATUS_ERROR && error != nullptr,
            "invalid struct version did not produce an owned error");
    require(error->struct_version == WAM_C_STRUCT_VERSION &&
                error->message != nullptr && error->details_json != nullptr,
            "C ABI error is missing its layout or payload");
    wam_c_error_free(error);

    wam::test::TempFile file("c-api-v4");
    wam::test::valid_gwp05_policy_fixture().write(file.string());
    const std::string artifact_path = file.string();
    wam_c_model_options_init(&options);
    options.artifact_path = artifact_path.c_str();
    options.backend = WAM_C_BACKEND_CPU_METADATA;
    require(wam_c_model_create(&options, &model, &error) == WAM_C_STATUS_OK &&
                model != nullptr && error == nullptr,
            "C ABI could not load a metadata-only GGUF");

    char * metadata = nullptr;
    require(wam_c_model_metadata_json(model, &metadata, &error) ==
                WAM_C_STATUS_OK && metadata != nullptr,
            "C ABI metadata query failed");
    require(std::strstr(metadata, "\"architecture\":\"gwp05\"") != nullptr,
            "C ABI metadata changed architecture");
    wam_c_string_free(metadata);

    wam_c_session_options session_options;
    wam_c_session_options_init(&session_options);
    wam_c_session * session = nullptr;
    require(wam_c_session_create(model, &session_options, &session, &error) ==
                WAM_C_STATUS_ERROR && session == nullptr && error != nullptr,
            "metadata backend unexpectedly created an execution session");
    wam_c_error_free(error);
    wam_c_session_free(nullptr);
    wam_c_prediction_free(nullptr);
    wam_c_model_free(model);
    wam_c_model_free(nullptr);
    return 0;
}
