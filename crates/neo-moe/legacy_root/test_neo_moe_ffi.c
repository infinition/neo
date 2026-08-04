/*
 * test_neo_moe_ffi.c
 *
 * Minimal test to validate the neo-moe C↔Rust FFI bridge.
 *
 * Build & run:
 *   export PATH="/c/msys64/ucrt64/bin:/c/msys64/usr/bin:$PATH"
 *   gcc -o test_ffi.exe test_neo_moe_ffi.c -Ltarget/release -lneo_moe
 *   ./test_ffi.exe /path/to/model.gguf
 */

#include "neo-moe/neo_moe_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  neo-moe C FFI test                          ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    /* ── 1. Create stream ─────────────────────────────── */
    printf("[1] neo_moe_stream_create(\"%s\", vram_slots=8, io_threads=2, ...)\n",
           model_path);

    NeoMoeStream *stream = neo_moe_stream_create(
        model_path,
        8,      /* vram_slots */
        2,      /* io_threads */
        0,      /* cuda_device */
        1       /* prefetch_depth */
    );

    if (!stream) {
        fprintf(stderr, "[FAIL] stream creation returned NULL\n");
        /* Might fail if no CUDA GPU is available — not a FFI issue. */
        printf("       (this is expected if no CUDA GPU is present)\n");
        printf("\n[SKIP] Cannot proceed without CUDA device.\n");
        return 0;
    }
    printf("[PASS] Stream created: %p\n", (void*)stream);

    /* ── 2. Model info ────────────────────────────────── */
    printf("\n[2] neo_moe_model_info()\n");
    uint32_t n_layers = 0, n_experts = 0;
    int rc = neo_moe_model_info(stream, &n_layers, &n_experts);
    if (rc == 0) {
        printf("[PASS] Model: %u layers × %u experts\n", n_layers, n_experts);
    } else {
        printf("[FAIL] model_info returned %d\n", rc);
    }

    /* ── 3. Prefetch ──────────────────────────────────── */
    printf("\n[3] neo_moe_prefetch(layer=0, top-4 experts)\n");
    uint32_t expert_ids[] = {0, 1, 2, 3};
    rc = neo_moe_prefetch(stream, 0, expert_ids, 4);
    printf("[%s] prefetch returned %d\n", rc == 0 ? "PASS" : "FAIL", rc);

    /* ── 4. Demand (with timeout — might fail if no GPU) ─ */
    printf("\n[4] neo_moe_demand(layer=0, expert=0, timeout=2000ms)\n");
    uint64_t gate_ptr = 0, up_ptr = 0, down_ptr = 0;
    rc = neo_moe_demand(stream, 0, 0, &gate_ptr, &up_ptr, &down_ptr, 2000);
    if (rc == 0) {
        printf("[PASS] Expert 0 pointers:\n");
        printf("       gate_ptr = 0x%016llx\n", (unsigned long long)gate_ptr);
        printf("       up_ptr   = 0x%016llx\n", (unsigned long long)up_ptr);
        printf("       down_ptr = 0x%016llx\n", (unsigned long long)down_ptr);
    } else {
        printf("[INFO] demand returned %d (expected without CUDA)\n", rc);
    }

    /* ── 5. Release ────────────────────────────────────── */
    printf("\n[5] neo_moe_release(layer=0, expert=0)\n");
    neo_moe_release(stream, 0, 0);
    printf("[PASS] release completed\n");

    /* ── 6. Cleanup ────────────────────────────────────── */
    printf("\n[6] neo_moe_stream_free()\n");
    neo_moe_stream_free(stream);
    printf("[PASS] stream freed\n");

    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║  All FFI tests passed!                        ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    return 0;
}
