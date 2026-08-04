#!/usr/bin/env bash
#
# integrate_llamacpp.sh
#
# Integrates neo-moe into a llama.cpp build tree.
#
# Usage:
#   cd /path/to/llama.cpp
#   bash /path/to/neo-moe/integrate_llamacpp.sh
#
# This will:
#   1. Copy neo_moe_backend.{c,h} into llama.cpp's source tree
#   2. Apply the neo-moe CMake patch (optional)
#   3. Print build instructions
#
# Environment:
#   NEO_MOE_DIR   — path to the neo-moe crate (default: ../neo-moe)
#   LLAMA_CPP_DIR — path to llama.cpp root       (default: .)
#   CUDA_TOOLKIT  — CUDA toolkit root             (default: auto-detect)

set -euo pipefail

NEO_MOE_DIR="${NEO_MOE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$(pwd)}"

echo "╔══════════════════════════════════════════════╗"
echo "║  neo-moe → llama.cpp integration             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  neo-moe:   ${NEO_MOE_DIR}"
echo "  llama.cpp: ${LLAMA_CPP_DIR}"
echo ""

# ── Step 1: Copy backend files ──────────────────────────────────────
echo "[1/4] Copying neo_moe_backend.* to ${LLAMA_CPP_DIR}/..."

cp -v "${NEO_MOE_DIR}/neo_moe_backend.h" "${LLAMA_CPP_DIR}/"
cp -v "${NEO_MOE_DIR}/neo_moe_backend.c" "${LLAMA_CPP_DIR}/"

# ── Step 2: Apply CMakeLists.txt patch ───────────────────────────────
echo ""
echo "[2/4] Applying CMakeLists.txt patch..."

CMAKE_FILE="${LLAMA_CPP_DIR}/CMakeLists.txt"
MARKER_1="# llama.cpp options"
MARKER_2="# Build static library"

# Add GGML_NEO_MOE option
if grep -q "GGML_NEO_MOE" "${CMAKE_FILE}" 2>/dev/null; then
    echo "  [SKIP] GGML_NEO_MOE already present in CMakeLists.txt"
else
    # Patch: add option after the GGML options block
    sed -i.bak '/option(GGML_SYCL_F16/a\
option(GGML_NEO_MOE   "llama: enable neo-moe MoE expert streaming" OFF)
' "${CMAKE_FILE}"

    # Patch for Windows (mingw): add link to neo_moe.dll
    # Patch for Linux: add link to libneo_moe.so
    echo "  [PATCH] Added GGML_NEO_MOE option"
fi

# ── Step 3: Create ggml-neo-moe.h header ────────────────────────────
# This header is included by the ggml backend to link neo-moe
echo ""
echo "[3/4] Creating ggml backend integration header..."

cat > "${LLAMA_CPP_DIR}/ggml-neo-moe.h" << 'HEADER'
/*
 * ggml-neo-moe.h
 *
 * Bridge between ggml and the neo-moe expert streaming engine.
 * Included by ggml-cuda.cu when GGML_NEO_MOE is enabled.
 *
 * Instead of loading all expert weight tensors into VRAM at model
 * load time, neo-moe streams only the active experts on-demand.
 */

#ifndef GGML_NEO_MOE_H
#define GGML_NEO_MOE_H

#include "neo_moe_backend.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Check if a tensor name corresponds to an MoE expert weight.
 * If so, mark it as managed by neo-moe (skip VRAM loading).
 *
 * Called during ggml_backend_cuda_buffer_type allocation.
 * Returns: 1 if managed by neo-moe, 0 otherwise.
 */
bool ggml_neo_moe_is_managed(const char *tensor_name);

/*
 * Register an expert tensor with neo-moe at model load time.
 * This records the tensor shape and location info without
 * loading the actual weight data into VRAM.
 */
int ggml_neo_moe_register_expert_tensor(
    const char *tensor_name,
    size_t      n_bytes
);

/*
 * Before a grouped expert matmul (build_lora_mm_id), ensure the
 * required expert weight slices are VRAM-resident.
 *
 * Returns 0 on success (tensors are ready for compute).
 */
int ggml_neo_moe_prepare_experts(
    int layer,
    const int *expert_ids,
    int n_experts
);

/*
 * After the expert matmul, release any temporaries.
 */
void ggml_neo_moe_finish_experts(int layer);

#ifdef __cplusplus
}
#endif

#endif /* GGML_NEO_MOE_H */
HEADER

echo "  [OK] ggml-neo-moe.h created"

# ── Step 4: Print build instructions ────────────────────────────────
echo ""
echo "[4/4] Build instructions"
echo ""

# Detect OS
UNAME_S=$(uname -s 2>/dev/null || echo "Windows")
case "${UNAME_S}" in
    Linux*)  OS="linux"  ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
    Darwin*) OS="macos"  ;;
    *)       OS="unknown" ;;
esac

echo "  OS detected: ${OS}"
echo ""

if [ "${OS}" = "windows" ]; then
    cat << 'WINBLD'
  Windows (MSYS2 UCRT64) build:

    # 1. Build neo-moe cdylib
    cd neo-moe
    cargo build --release

    # 2. Build llama.cpp with neo-moe (CPU only, no CUDA toolkit)
    cd ../llama.cpp
    mkdir -p build && cd build
    cmake .. -G "Ninja" \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DGGML_NEO_MOE=ON \
        -DLLAMA_CUDA=OFF
    cmake --build . --target llama-server

    # 3. Run
    export PATH="/c/msys64/ucrt64/bin:/path/to/neo-moe/target/release:$PATH"
    ./bin/llama-server --model model.gguf

WINBLD
elif [ "${OS}" = "linux" ]; then
    cat << 'LNXBLD'
  Linux build (with CUDA):

    # 1. Build neo-moe cdylib
    cd neo-moe
    cargo build --release

    # 2. Build llama.cpp with CUDA + neo-moe
    cd ../llama.cpp
    mkdir -p build && cd build
    cmake .. -G Ninja \
        -DGGML_CUDA=ON \
        -DGGML_NEO_MOE=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target llama-server -- -j$(nproc)

    # 3. Run with LD_PRELOAD or rpath
    LD_LIBRARY_PATH=/path/to/neo-moe/target/release:$LD_LIBRARY_PATH \
        ./bin/llama-server --model model.gguf

LNXBLD
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Integration complete!                        ║"
echo "║  Next: install CUDA toolkit, rebuild with    ║"
echo "║  -DGGML_CUDA=ON -DGGML_NEO_MOE=ON            ║"
echo "╚══════════════════════════════════════════════╝"
