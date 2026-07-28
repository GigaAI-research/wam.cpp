#include "fixtures/metadata_fixture.h"

#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: write-cli-fixture OUTPUT_GGUF\n";
        return 2;
    }
    wam::test::MetadataFixture fixture =
        wam::test::valid_gwp05_policy_fixture();
    fixture.write(argv[1]);
    return 0;
}
