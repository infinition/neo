#!/bin/bash
# Neo vs FFmpeg benchmark — 4K image filter passes.
#
# Override paths with env vars:
#   FFMPEG=/path/to/ffmpeg.exe NEO=./target/release/neo.exe INPUT=clip.png ./scripts/benchmark.sh

FFMPEG="${FFMPEG:-ffmpeg}"
NEO="${NEO:-./target/release/neo.exe}"
INPUT="${INPUT:-assets/videos/bench_4k_frame.png}"

echo ""
echo "================================================================"
echo "   NEO vs FFMPEG -- Benchmark Comparatif"
echo "   Image: 4096x1714 (4K DCP -- Tears of Steel)"
echo "================================================================"
echo ""

# -------------------------------------------------------
# TEST 1: Grayscale conversion
# -------------------------------------------------------
echo "--- TEST 1: Grayscale Conversion (4K) ---"
echo ""

echo "[FFmpeg] grayscale..."
START=$(date +%s%N)
"$FFMPEG" -i "$INPUT" -vf "format=gray" -update 1 -y ffmpeg_gray.png 2>/dev/null
END=$(date +%s%N)
FFMPEG_GRAY=$(( (END - START) / 1000000 ))
echo "  FFmpeg:     ${FFMPEG_GRAY} ms"

echo "[Neo] grayscale..."
START=$(date +%s%N)
$NEO process -i "$INPUT" -o neo_gray.png -f grayscale 2>/dev/null
END=$(date +%s%N)
NEO_GRAY=$(( (END - START) / 1000000 ))
echo "  Neo: ${NEO_GRAY} ms"
echo ""

# -------------------------------------------------------
# TEST 2: Blur
# -------------------------------------------------------
echo "--- TEST 2: Blur (4K) ---"
echo ""

echo "[FFmpeg] boxblur..."
START=$(date +%s%N)
"$FFMPEG" -i "$INPUT" -vf "boxblur=2:2" -update 1 -y ffmpeg_blur.png 2>/dev/null
END=$(date +%s%N)
FFMPEG_BLUR=$(( (END - START) / 1000000 ))
echo "  FFmpeg:     ${FFMPEG_BLUR} ms"

echo "[Neo] blur..."
START=$(date +%s%N)
$NEO process -i "$INPUT" -o neo_blur.png -f blur 2>/dev/null
END=$(date +%s%N)
NEO_BLUR=$(( (END - START) / 1000000 ))
echo "  Neo: ${NEO_BLUR} ms"
echo ""

# -------------------------------------------------------
# TEST 3: Edge detection
# -------------------------------------------------------
echo "--- TEST 3: Edge Detection (4K) ---"
echo ""

echo "[FFmpeg] edgedetect..."
START=$(date +%s%N)
"$FFMPEG" -i "$INPUT" -vf "edgedetect=low=0.1:high=0.3" -update 1 -y ffmpeg_edge.png 2>/dev/null
END=$(date +%s%N)
FFMPEG_EDGE=$(( (END - START) / 1000000 ))
echo "  FFmpeg:     ${FFMPEG_EDGE} ms"

echo "[Neo] edge-detect..."
START=$(date +%s%N)
$NEO process -i "$INPUT" -o neo_edge.png -f edge-detect 2>/dev/null
END=$(date +%s%N)
NEO_EDGE=$(( (END - START) / 1000000 ))
echo "  Neo: ${NEO_EDGE} ms"
echo ""

# -------------------------------------------------------
# TEST 4: Sharpen
# -------------------------------------------------------
echo "--- TEST 4: Sharpen (4K) ---"
echo ""

echo "[FFmpeg] unsharp..."
START=$(date +%s%N)
"$FFMPEG" -i "$INPUT" -vf "unsharp=5:5:1.5" -update 1 -y ffmpeg_sharp.png 2>/dev/null
END=$(date +%s%N)
FFMPEG_SHARP=$(( (END - START) / 1000000 ))
echo "  FFmpeg:     ${FFMPEG_SHARP} ms"

echo "[Neo] sharpen..."
START=$(date +%s%N)
$NEO process -i "$INPUT" -o neo_sharp.png -f sharpen 2>/dev/null
END=$(date +%s%N)
NEO_SHARP=$(( (END - START) / 1000000 ))
echo "  Neo: ${NEO_SHARP} ms"
echo ""

# -------------------------------------------------------
# TEST 5: Scale 2x (upscale)
# -------------------------------------------------------
echo "--- TEST 5: Upscale 2x (4K -> 8K) ---"
echo ""

echo "[FFmpeg] scale 2x (bilinear)..."
START=$(date +%s%N)
"$FFMPEG" -i "$INPUT" -vf "scale=iw*2:ih*2:flags=bilinear" -update 1 -y ffmpeg_upscale.png 2>/dev/null
END=$(date +%s%N)
FFMPEG_UP=$(( (END - START) / 1000000 ))
echo "  FFmpeg:     ${FFMPEG_UP} ms"

echo "[Neo] upscale-2x..."
START=$(date +%s%N)
$NEO process -i "$INPUT" -o neo_upscale.png -f upscale-2x 2>/dev/null
END=$(date +%s%N)
NEO_UP=$(( (END - START) / 1000000 ))
echo "  Neo: ${NEO_UP} ms"
echo ""

# -------------------------------------------------------
# TEST 6: Chain of 3 filters
# -------------------------------------------------------
echo "--- TEST 6: Filter Chain (grayscale + sharpen + blur) ---"
echo ""

echo "[FFmpeg] chain..."
START=$(date +%s%N)
"$FFMPEG" -i "$INPUT" -vf "format=gray,unsharp=5:5:1.5,boxblur=2:2" -update 1 -y ffmpeg_chain.png 2>/dev/null
END=$(date +%s%N)
FFMPEG_CHAIN=$(( (END - START) / 1000000 ))
echo "  FFmpeg:     ${FFMPEG_CHAIN} ms"

echo "[Neo] chain..."
START=$(date +%s%N)
$NEO process -i "$INPUT" -o neo_chain.png -f grayscale,sharpen,blur 2>/dev/null
END=$(date +%s%N)
NEO_CHAIN=$(( (END - START) / 1000000 ))
echo "  Neo: ${NEO_CHAIN} ms"
echo ""

# -------------------------------------------------------
# RESULTS
# -------------------------------------------------------
echo "================================================================"
echo "   RESULTS SUMMARY -- 4096x1714 (4K DCP)"
echo "================================================================"
echo ""
printf "  %-20s %12s %12s %10s\n" "Filter" "FFmpeg" "Neo" "Speedup"
printf "  %-20s %12s %12s %10s\n" "------" "------" "----------" "-------"
printf "  %-20s %10d ms %10d ms %8.1fx\n" "Grayscale" "$FFMPEG_GRAY" "$NEO_GRAY" "$(echo "scale=1; $FFMPEG_GRAY / $NEO_GRAY" | bc)"
printf "  %-20s %10d ms %10d ms %8.1fx\n" "Blur" "$FFMPEG_BLUR" "$NEO_BLUR" "$(echo "scale=1; $FFMPEG_BLUR / $NEO_BLUR" | bc)"
printf "  %-20s %10d ms %10d ms %8.1fx\n" "Edge Detect" "$FFMPEG_EDGE" "$NEO_EDGE" "$(echo "scale=1; $FFMPEG_EDGE / $NEO_EDGE" | bc)"
printf "  %-20s %10d ms %10d ms %8.1fx\n" "Sharpen" "$FFMPEG_SHARP" "$NEO_SHARP" "$(echo "scale=1; $FFMPEG_SHARP / $NEO_SHARP" | bc)"
printf "  %-20s %10d ms %10d ms %8.1fx\n" "Upscale 2x" "$FFMPEG_UP" "$NEO_UP" "$(echo "scale=1; $FFMPEG_UP / $NEO_UP" | bc)"
printf "  %-20s %10d ms %10d ms %8.1fx\n" "3-filter Chain" "$FFMPEG_CHAIN" "$NEO_CHAIN" "$(echo "scale=1; $FFMPEG_CHAIN / $NEO_CHAIN" | bc)"
echo ""
echo "  FFmpeg 8.1 (CPU) vs Neo (RTX 4070 Ti, Vulkan)"
echo "  Image: Tears of Steel 4K DCP frame (4096x1714)"
echo "================================================================"
