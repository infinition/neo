/*
 * neo_moe_backend.h
 *
 * Bridge between llama.cpp/ggml and the neo-moe MoE expert streaming engine.
 *
 * Integration points:
 *   1. Call ggml_neo_moe_init() after model loading (provides GGUF path)
 *   2. ggml_neo_moe_is_expert_tensor() marks expert tensors at load time
 *   3. ggml_neo_moe_prepare_experts() before grouped expert matmul
 *   4. ggml_neo_moe_shutdown() at exit
 *
 * BUILD
 * -----
 *   In llama.cpp build tree with -DGGML_NEO_MOE=ON:
 *     cmake -DGGML_NEO_MOE=ON -DNEO_MOE_LIB_DIR=/path/to/neo-moe/release ..
 *     cmake --build .
 *
 *   The build system automatically includes neo_moe_backend.c in the
 *   llama library and links against neo_moe.dll / libneo_moe.so.
 *
 * STANDALONE TEST (no llama.cpp)
 *   gcc -o test_neo_moe_ffi.exe test_neo_moe_ffi.c -I. \
 *       -L/path/to/neo-moe/target/release -lneo_moe
 */

#ifndef NEO_MOE_BACKEND_H
#define NEO_MOE_BACKEND_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ══════════════════════════════════════════════════════════════════════════════*/

/*
 * Initialize the neo-moe streaming engine.
 * Call once after model loading, before the first inference pass.
 *
 * gguf_path: path to the GGUF model file (UTF-8).
 * Returns 0 on success, negative on failure.
 */
int  ggml_neo_moe_init(const char *gguf_path);

/*
 * Shutdown and free all neo-moe resources.
 */
void ggml_neo_moe_shutdown(void);

/* ════════════════════════════════════════════════════════════════════════════════
 * Expert tensor management
 * ══════════════════════════════════════════════════════════════════════════════*/

/*
 * Check if a tensor name corresponds to an MoE expert weight tensor.
 * If true, the tensor should be registered with neo-moe (skip VRAM loading).
 *
 * Matches naming patterns:
 *   blk.{L}.ffn_gate_exps.{E}.weight   (Qwen3, Mixtral)
 *   blk.{L}.ffn_up_exps.{E}.weight
 *   blk.{L}.ffn_down_exps.{E}.weight
 *   blk.{L}.ffn_gate_up_exps.weight    (fused gate+up)
 */
bool ggml_neo_moe_is_expert_tensor(const char *tensor_name);

/* ════════════════════════════════════════════════════════════════════════════════
 * Inference hooks
 * ══════════════════════════════════════════════════════════════════════════════*/

/*
 * Prepare experts for a MoE layer.
 * Called before build_lora_mm_id() / grouped expert matmul.
 * This prefetches the required expert weight slices from NVMe → VRAM.
 *
 * layer:      transformer layer index
 * expert_ids: array of active expert indices for this layer
 * n_experts:  number of entries in expert_ids
 *
 * Returns 0 on success (or if experts are already VRAM-resident).
 */
int  ggml_neo_moe_prepare_experts(int layer, const int *expert_ids, int n_experts);

/*
 * Cleanup after the MoE layer computation.
 * Currently a no-op (cleanup is reference-counted).
 */
void ggml_neo_moe_finish_experts(int layer);

/* ════════════════════════════════════════════════════════════════════════════════
 * Test / validation
 * ══════════════════════════════════════════════════════════════════════════════*/

/*
 * Run a quick FFI bridge validation test.
 * Creates a stream, queries model info, prefetches + demands + releases.
 * Returns 0 on success, negative on failure.
 */
int ggml_neo_moe_test_ffi(const char *gguf_path);

#ifdef __cplusplus
}
#endif

#endif /* NEO_MOE_BACKEND_H */
