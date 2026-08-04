/*
 * test_neo_moe_runtime.c
 *
 * Runtime test: verify that neo-moe FFI works through the
 * ggml_neo_moe_test_ffi() function compiled into llama library.
 *
 * Build & run:
 *   gcc -o test_runtime.exe test_neo_moe_runtime.c \
 *       -I/path/to/llama.cpp \
 *       /path/to/llama.cpp/build_neo_moe/src/libllama.a \
 *       /path/to/llama.cpp/build_neo_moe/ggml/src/ggml.a \
 *       /path/to/llama.cpp/build_neo_moe/ggml/src/ggml-cpu.a \
 *       /path/to/llama.cpp/build_neo_moe/ggml/src/ggml-base.a \
 *       -fopenmp -lws2_32 -lneo_moe
 *
 *   export PATH="/path/to/neo-moe/target/release:$PATH"
 *   ./test_runtime.exe /path/to/model.gguf
 */

#include "neo_moe_backend.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    printf("[neo-moe] Runtime FFI test\n");
    printf("[neo-moe] Running ggml_neo_moe_test_ffi(\"%s\")...\n\n", argv[1]);

    int rc = ggml_neo_moe_test_ffi(argv[1]);

    printf("\n[neo-moe] Test %s (rc=%d)\n",
           rc == 0 ? "PASSED" : "FAILED", rc);

    return rc;
}
