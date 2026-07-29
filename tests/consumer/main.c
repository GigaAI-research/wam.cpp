#include <wam/c_api.h>

int main(void) {
    wam_c_model_options options;
    wam_c_model_options_init(&options);
    return wam_c_abi_version() == WAM_C_ABI_VERSION &&
            options.struct_version == WAM_C_STRUCT_VERSION ? 0 : 1;
}
