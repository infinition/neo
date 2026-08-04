/*
 * neo_moe_backend.c
 *
 * Custom ggml_backend that injects neo-moe VRAM slots directly into
 * llama.cpp's tensor allocation path, bypassing the default CUDA allocator
 * for MoE expert weight tensors.
 *
 * HOW TO USE
 * ----------
 * 1. Drop this file next to llama.cpp's ggml-cuda.cu (or anywhere in the build).
 * 2. Apply the minimal patch to ggml.c (see patch/ggml_dispatch.patch).
 * 3. Link your inference binary against both llama.cpp and neo-moe-ffi.
 *
 * INTEGRATION FLOW
 * ----------------
 *
 *  llama.cpp startup                neo-moe
 *  ─────────────────────────────   ──────────────────────────────────
 *  llama_model_load()
 *    ggml_backend_buffer_alloc()
 *      → neo_moe_buf_alloc()  ──── weight_map lookup
 *                                    ↳ expert tensor? → return NULL sentinel
 *                                    ↳ non-expert?    → defer to CUDA allocator
 *
 *  token forward pass
 *    moe_dispatch():
 *      router_logits → neo_moe_prefetch()
 *      for each active expert:
 *        neo_moe_demand()           ← blocks only on cache miss (~1.7 ms)
 *        inject_expert_ptr()        ← swaps ggml_tensor.data in-place
 *        ggml_cuda_op_mul_mat()     ← runs in VRAM, no copy
 *        neo_moe_release()
 *
 * PATCHING STRATEGY
 * -----------------
 * Rather than forking llama.cpp, we intercept at two call sites:
 *
 *   a) `ggml_backend_buft_alloc_buffer` in ggml-backend.c:
 *      We register our buffer type as the allocator for expert tensors.
 *
 *   b) `llm_build_moe_ffn` in llama.cpp:
 *      We add three lines that call neo_moe_demand() + inject_expert_ptr()
 *      before each ggml_mul_mat call, and neo_moe_release() after.
 *
 * See patch/ggml_dispatch.patch for the exact diff.
 */

#include "neo_moe_backend.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ──────────────────────────────────────────────────────────────────────────────
 * FFI declarations — implemented in neo-moe-ffi (Rust cdylib)
 * ────────────────────────────────────────────────────────────────────────────*/

/* Opaque handle returned by neo_moe_stream_create(). */
typedef void* NeoMoeStream;

/* Demand expert (layer, expert_id) → fills *gate_ptr, *up_ptr, *down_ptr.
 * Blocks until the expert is VRAM-resident (max timeout_ms milliseconds).
 * Returns 0 on success, negative errno on failure.                          */
extern int neo_moe_demand(
    NeoMoeStream stream,
    uint32_t     layer,
    uint32_t     expert_id,
    uint64_t    *gate_ptr,    /* out: raw CUDA device pointer */
    uint64_t    *up_ptr,
    uint64_t    *down_ptr,
    uint32_t     timeout_ms
);

/* Submit speculative prefetch requests (non-blocking).                       */
extern int neo_moe_prefetch(
    NeoMoeStream stream,
    uint32_t     layer,
    const uint32_t *expert_ids,
    uint32_t     n_experts
);

/* Release the VRAM slot for (layer, expert_id).                              */
extern void neo_moe_release(
    NeoMoeStream stream,
    uint32_t     layer,
    uint32_t     expert_id
);

/* ──────────────────────────────────────────────────────────────────────────────
 * Global stream handle — set once after llama_model_load()
 * ────────────────────────────────────────────────────────────────────────────*/

static NeoMoeStream g_stream = NULL;

void neo_moe_set_stream(NeoMoeStream stream) {
    g_stream = stream;
}

NeoMoeStream neo_moe_get_stream(void) {
    return g_stream;
}

/* ──────────────────────────────────────────────────────────────────────────────
 * Expert tensor name detection
 *
 * Returns 1 if `name` matches an expert weight tensor that neo-moe manages,
 * and fills *layer_out / *expert_out.
 * Pattern: "blk.{L}.ffn_{gate|up|down}_exps.{E}.weight"
 * ────────────────────────────────────────────────────────────────────────────*/

int neo_moe_is_expert_tensor(
    const char *name,
    uint32_t   *layer_out,
    uint32_t   *expert_out
) {
    unsigned int layer, expert;
    char proj[32];

    /* Qwen3 / Mixtral naming (llama.cpp convention). */
    int n = sscanf(name, "blk.%u.ffn_%31[^.].%u.weight", &layer, proj, &expert);
    if (n != 3) return 0;

    /* Only intercept the three expert projection types. */
    if (strncmp(proj, "gate_exps", 9) != 0 &&
        strncmp(proj, "up_exps",   7) != 0 &&
        strncmp(proj, "down_exps", 9) != 0 &&
        strncmp(proj, "gate_exp",  8) != 0 &&   /* Mixtral singular */
        strncmp(proj, "up_exp",    6) != 0 &&
        strncmp(proj, "down_exp",  8) != 0)
        return 0;

    if (layer_out)  *layer_out  = (uint32_t)layer;
    if (expert_out) *expert_out = (uint32_t)expert;
    return 1;
}

/* ──────────────────────────────────────────────────────────────────────────────
 * inject_expert_ptr
 *
 * Swaps `tensor->data` with the CUDA device pointer from the neo-moe pool,
 * so that the subsequent ggml_cuda_op_mul_mat() operates directly on the
 * streamed weights — zero copy from VRAM pool slot to matmul input.
 *
 * Called from the patched llm_build_moe_ffn() in llama.cpp.
 * ────────────────────────────────────────────────────────────────────────────*/

int neo_moe_inject_expert(
    struct ggml_tensor *gate_tensor,
    struct ggml_tensor *up_tensor,
    struct ggml_tensor *down_tensor,
    uint32_t            layer,
    uint32_t            expert_id,
    uint32_t            timeout_ms
) {
    if (!g_stream) {
        fprintf(stderr, "[neo-moe] ERROR: stream not initialised\n");
        return -1;
    }

    uint64_t gate_ptr = 0, up_ptr = 0, down_ptr = 0;

    int rc = neo_moe_demand(
        g_stream, layer, expert_id,
        &gate_ptr, &up_ptr, &down_ptr,
        timeout_ms
    );
    if (rc != 0) {
        fprintf(stderr,
            "[neo-moe] demand failed for layer=%u expert=%u (rc=%d)\n",
            layer, expert_id, rc);
        return rc;
    }

    /* Swap data pointers in-place.  ggml_tensor.data is a void* so the cast
     * is safe; the GPU kernel reads from this address directly.             */
    if (gate_tensor)  gate_tensor->data  = (void*)(uintptr_t)gate_ptr;
    if (up_tensor)    up_tensor->data    = (void*)(uintptr_t)up_ptr;
    if (down_tensor)  down_tensor->data  = (void*)(uintptr_t)down_ptr;

    return 0;
}

void neo_moe_release_expert(uint32_t layer, uint32_t expert_id) {
    if (g_stream) {
        neo_moe_release(g_stream, layer, expert_id);
    }
}

/* ──────────────────────────────────────────────────────────────────────────────
 * ggml custom backend buffer type
 *
 * Registered as the buffer type for expert tensors at model-load time.
 * When llama.cpp calls ggml_backend_buft_alloc_buffer() for an expert tensor,
 * we return a sentinel buffer whose `data` pointer is NULL — the real pointer
 * is injected later by neo_moe_inject_expert().
 *
 * For non-expert tensors we fall through to the default CUDA allocator.
 * ────────────────────────────────────────────────────────────────────────────*/

/* Sentinel buffer: no real allocation, just a marker. */
typedef struct {
    uint32_t layer;
    uint32_t expert;
    int      proj;   /* 0=gate, 1=up, 2=down */
} NeoMoeBufCtx;

static const char * neo_buft_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "NEO_MOE";
}

static ggml_backend_buffer_t neo_buft_alloc_buffer(
    ggml_backend_buffer_type_t buft,
    size_t size
) {
    /* Return a zero-sized buffer — data pointer will be swapped per-token. */
    (void)buft; (void)size;
    ggml_backend_buffer_t buf = ggml_backend_buffer_init(
        buft,
        &neo_buffer_i,   /* interface, defined below */
        NULL,            /* context: no allocation   */
        0                /* size: zero               */
    );
    return buf;
}

static size_t neo_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 512; /* sector alignment, matches O_DIRECT contract */
}

static bool neo_buft_supports_backend(
    ggml_backend_buffer_type_t buft,
    ggml_backend_t             backend
) {
    (void)buft;
    /* We support the CUDA backend only. */
    return ggml_backend_is_cuda(backend);
}

static bool neo_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return false; /* data lives in VRAM */
}

static struct ggml_backend_buffer_type_i neo_buft_i = {
    /* .get_name           = */ neo_buft_name,
    /* .alloc_buffer       = */ neo_buft_alloc_buffer,
    /* .get_alignment      = */ neo_buft_get_alignment,
    /* .get_max_size       = */ NULL,
    /* .get_alloc_size     = */ NULL,
    /* .supports_backend   = */ neo_buft_supports_backend,
    /* .is_host            = */ neo_buft_is_host,
};

/* ── Buffer interface (called by ggml on individual tensor ops) ──────────── */

static void neo_buf_free_buffer(ggml_backend_buffer_t buf) {
    (void)buf; /* nothing to free — slots are owned by neo-moe pool */
}

static void * neo_buf_get_base(ggml_backend_buffer_t buf) {
    (void)buf;
    return NULL; /* real pointer injected at forward-pass time */
}

static void neo_buf_init_tensor(ggml_backend_buffer_t buf, struct ggml_tensor *tensor) {
    /* Mark tensor so we can detect it in llm_build_moe_ffn.
     * We store a non-null extra pointer as a sentinel flag.               */
    (void)buf;
    tensor->extra = (void*)(uintptr_t)0xNEO0; /* magic tag */
}

static void neo_buf_set_tensor(
    ggml_backend_buffer_t buf,
    struct ggml_tensor *tensor,
    const void *data,
    size_t offset,
    size_t size
) {
    /* Called during model load to copy weights from host into the "buffer".
     * We discard the data — neo-moe reads directly from the GGUF file.    */
    (void)buf; (void)tensor; (void)data; (void)offset; (void)size;
    /* No-op: weights are NOT loaded into VRAM at model-load time.          */
}

static void neo_buf_get_tensor(
    ggml_backend_buffer_t buf,
    const struct ggml_tensor *tensor,
    void *data,
    size_t offset,
    size_t size
) {
    (void)buf; (void)tensor; (void)data; (void)offset; (void)size;
    /* Not needed for inference. */
}

static bool neo_buf_cpy_tensor(
    ggml_backend_buffer_t buf,
    const struct ggml_tensor *src,
    struct ggml_tensor *dst
) {
    (void)buf; (void)src; (void)dst;
    return false; /* handled by the GPU path */
}

static void neo_buf_clear(ggml_backend_buffer_t buf, uint8_t val) {
    (void)buf; (void)val;
    /* Nothing to clear — slots are managed by neo-moe pool. */
}

static struct ggml_backend_buffer_i neo_buffer_i = {
    /* .free_buffer  = */ neo_buf_free_buffer,
    /* .get_base     = */ neo_buf_get_base,
    /* .init_tensor  = */ neo_buf_init_tensor,
    /* .memset_tensor= */ neo_buf_clear,
    /* .set_tensor   = */ neo_buf_set_tensor,
    /* .get_tensor   = */ neo_buf_get_tensor,
    /* .cpy_tensor   = */ neo_buf_cpy_tensor,
    /* .synchronize  = */ NULL,
};

/* ── Public: return our buffer type for expert tensors ─────────────────── */

static struct ggml_backend_buffer_type neo_buft = {
    /* .iface  = */ &neo_buft_i,
    /* .device = */ NULL,
    /* .context= */ NULL,
};

ggml_backend_buffer_type_t neo_moe_buffer_type(void) {
    return &neo_buft;
}
