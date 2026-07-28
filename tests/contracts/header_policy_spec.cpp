#include "wam/policy_spec.h"

int main() {
    return wam::kPolicySpecSchemaVersion == 2 ? 0 : 1;
}
