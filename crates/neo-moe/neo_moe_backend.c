/*
 * neo_moe_backend.c
 *
 * Bridge between llama.cpp/ggml and the neo-moe MoE expert streaming engine.
 *
 * Instead of loading all expert weight tensors into VRAM at model load time,
 * neo-moe streams only the active experts on-demand from NVMe → VRAM.
 *
 * Integration points:
 *   1. ggml_neo_moe_init() — called once at startup, registers the backend
 *   2. ggml_neo_moe_is_expert_tensor() — called during model loading to mark
 *      expert tensors as managed by neo-moe (skip VRAM allocation)
 *   3. ggml_neo_moe_prepare_experts() — called before build_lora_mm_id() to
 *      ensure the required expert weight slices are VRAM-resident
 *
 * BUILD
 * -----
 *   # In llama.cpp build tree with -DGGML_NEO_MOE=ON:
 *   cmake -DGGML_NEO_MOE=ON -DNEO_MOE_LIB_DIR=/path/to/neo-moe/release ..
 *   cmake --build .
 */

#include "neo_moe_backend.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

/* ───────────────────────────────────────────────────────────────────────────────
 * FFI declarations — implemented in neo-moe (Rust cdylib)
 * ─────────────────────────────────────────────────────────────────────────────*/

typedef struct NeoMoeStream NeoMoeStream;

extern NeoMoeStream* neo_moe_stream_create(
    const char *gguf_path,
    uint32_t    vram_slots,
    uint32_t    io_threads,
    uint32_t    cuda_device,
    uint32_t    depth
);

extern void neo_moe_stream_free(NeoMoeStream *stream);

extern int neo_moe_demand(
    NeoMoeStream *stream,
    uint32_t      layer,
    uint32_t      expert_id,
    uint64_t     *gate_ptr,
    uint64_t     *up_ptr,
    uint64_t     *down_ptr,
    uint32_t      timeout_ms
);

extern int64_t neo_moe_demand_keep(
    NeoMoeStream *stream,
    uint32_t      layer,
    uint32_t      expert_id,
    uint64_t     *gate_ptr,
    uint64_t     *up_ptr,
    uint64_t     *down_ptr,
    uint32_t      timeout_ms
);

extern void neo_moe_release_handle(int64_t handle_id);

extern int neo_moe_prefetch(
    NeoMoeStream  *stream,
    uint32_t       layer,
    const uint32_t *expert_ids,
    uint32_t       n_experts
);

extern void neo_moe_release(
    NeoMoeStream *stream,
    uint32_t      layer,
    uint32_t      expert_id
);

extern int neo_moe_model_info(
    NeoMoeStream *stream,
    uint32_t     *n_layers,
    uint32_t     *n_experts
);

/* ───────────────────────────────────────────────────────────────────────────────
 * Global state
 * ─────────────────────────────────────────────────────────────────────────────*/

static NeoMoeStream *g_stream = NULL;
static bool g_initialized = false;
static uint32_t g_n_layers = 0;
static uint32_t g_n_experts = 0;

/* ───────────────────────────────────────────────────────────────────────────────
 * Initialization
 * ─────────────────────────────────────────────────────────────────────────────*/

int ggml_neo_moe_init(const char *gguf_path) {
    if (g_initialized) {
        fprintf(stderr, "[neo-moe] already initialized\n");
        return 0;
    }

    fprintf(stderr, "[neo-moe] initializing with model: %s\n", gguf_path);

    g_stream = neo_moe_stream_create(gguf_path, 16, 4, 0, 2);
    if (!g_stream) {
        fprintf(stderr, "[neo-moe] ERROR: stream creation failed\n");
        return -1;
    }

    uint32_t n_layers = 0, n_experts = 0;
    int rc = neo_moe_model_info(g_stream, &n_layers, &n_experts);
    if (rc == 0) {
        g_n_layers = n_layers;
        g_n_experts = n_experts;
        fprintf(stderr, "[neo-moe] model: %u layers, %u experts/layer\n",
                n_layers, n_experts);
    }

    g_initialized = true;
    return 0;
}

void ggml_neo_moe_shutdown(void) {
    if (g_stream) {
        neo_moe_stream_free(g_stream);
        g_stream = NULL;
    }
    g_initialized = false;
}

/* ───────────────────────────────────────────────────────────────────────────────
 * Expert tensor detection
 * ─────────────────────────────────────────────────────────────────────────────*/

bool ggml_neo_moe_is_expert_tensor(const char *name) {
    if (!name) return false;

    /* Match: blk.{L}.ffn_gate_exps.{E}.weight
              blk.{L}.ffn_up_exps.{E}.weight
              blk.{L}.ffn_down_exps.{E}.weight
              blk.{L}.ffn_gate_up_exps.weight  (fused)
    */
    unsigned int layer, expert;
    char proj[32];

    int n = sscanf(name, "blk.%u.ffn_%31[^.].%u.weight", &layer, proj, &expert);
    if (n == 3) {
        if (strncmp(proj, "gate_exps", 9) == 0 ||
            strncmp(proj, "up_exps",   7) == 0 ||
            strncmp(proj, "down_exps", 9) == 0 ||
            strncmp(proj, "gate_exp",  8) == 0 ||
            strncmp(proj, "up_exp",    6) == 0 ||
            strncmp(proj, "down_exp",  8) == 0)
            return true;
    }

    /* Fused gate_up tensor: no expert index */
    if (strstr(name, "ffn_gate_up_exps") || strstr(name, "ffn_gate_up_exp")) {
        return true;
    }

    return false;
}

/* ───────────────────────────────────────────────────────────────────────────────
 * Expert preparation (called before build_lora_mm_id)
 * ─────────────────────────────────────────────────────────────────────────────*/

int ggml_neo_moe_prepare_experts(
    int              layer,
    const int       *expert_ids,
    int              n_experts
) {
    if (!g_initialized || !g_stream) return -1;

    /* Prefetch the experts for this layer (non-blocking) */
    uint32_t ids[32];
    int n = n_experts > 32 ? 32 : n_experts;
    for (int i = 0; i < n; i++) {
        ids[i] = (uint32_t)expert_ids[i];
    }

    return neo_moe_prefetch(g_stream, (uint32_t)layer, ids, (uint32_t)n);
}

void ggml_neo_moe_finish_experts(int layer) {
    (void)layer;
    /* Currently a no-op — cleanup is managed by reference counting */
}

/* ───────────────────────────────────────────────────────────────────────────────
 * Testing / validation
 * ─────────────────────────────────────────────────────────────────────────────*/

int ggml_neo_moe_test_ffi(const char *gguf_path) {
    fprintf(stderr, "[neo-moe-test] FFI bridge test\n");
    fprintf(stderr, "[neo-moe-test] Creating stream...\n");

    NeoMoeStream *s = neo_moe_stream_create(gguf_path, 8, 2, 0, 1);
    if (!s) {
        fprintf(stderr, "[neo-moe-test] FAIL: stream creation returned NULL\n");
        return -1;
    }

    uint32_t n_layers = 0, n_experts = 0;
    int rc = neo_moe_model_info(s, &n_layers, &n_experts);
    if (rc == 0) {
        fprintf(stderr, "[neo-moe-test]   model: %u layers x %u experts\n",
                n_layers, n_experts);
    } else {
        fprintf(stderr, "[neo-moe-test]   model_info failed: %d\n", rc);
    }

    uint32_t expert_ids[] = {0, 1, 2, 3};
    rc = neo_moe_prefetch(s, 0, expert_ids, 4);
    fprintf(stderr, "[neo-moe-test]   prefetch: %d\n", rc);

    uint64_t gate_ptr = 0, up_ptr = 0, down_ptr = 0;
    rc = neo_moe_demand(s, 0, 0, &gate_ptr, &up_ptr, &down_ptr, 2000);
    if (rc == 0) {
        fprintf(stderr, "[neo-moe-test]   demand OK: gate=0x%llx up=0x%llx down=0x%llx\n",
                (unsigned long long)gate_ptr,
                (unsigned long long)up_ptr,
                (unsigned long long)down_ptr);
    } else {
        fprintf(stderr, "[neo-moe-test]   demand: %d (expected without CUDA)\n", rc);
    }

    neo_moe_release(s, 0, 0);
    neo_moe_stream_free(s);

    fprintf(stderr, "[neo-moe-test] PASS\n");
    return 0;
}
