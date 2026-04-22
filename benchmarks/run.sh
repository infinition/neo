#!/usr/bin/env bash
#
# Neo benchmark runner.
#
# Runs a fixed clip through every Neo backend + a few FFmpeg reference
# configurations, repeats each config N times, and emits a CSV that the
# companion `render.sh` script turns into markdown tables.
#
# Requirements:
#   - neo.exe built in release mode (target/release/)
#   - ffmpeg + ffprobe on PATH with h264_nvenc support
#   - bash 4+, GNU coreutils (date +%s.%N), awk
#
# Run from the neo repo root:
#   ./benchmarks/run.sh
#

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=${RUNS:-5}
RESULTS=benchmarks/results.csv
NEO=./target/release/neo.exe
TMPOUT=/tmp/neo-bench-out

mkdir -p "$(dirname "$TMPOUT")"
echo "backend,resolution,run,wall_s,frames,bytes" >"$RESULTS"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "$NEO" ]] || die "neo.exe not found at $NEO (cargo build --release -p neo-cli)"
command -v ffmpeg >/dev/null || die "ffmpeg not on PATH"
command -v ffprobe >/dev/null || die "ffprobe not on PATH"

# Clock helper (seconds with fractional precision).
now() { date +%s.%N; }

# Count decoded frames in an H.264 file.
count_frames() {
    ffprobe -v error -count_frames -show_entries stream=nb_read_frames \
        -select_streams v:0 "$1" 2>/dev/null |
        awk -F= '/nb_read_frames/ {print $2}'
}

run_neo() {
    local backend=$1 res=$2 in_file=$3 run=$4
    local out="${TMPOUT}.${backend}.${res}.h264"
    local t0 t1
    t0=$(now)
    "$NEO" transcode-test -i "$in_file" -o "$out" --backend "$backend" \
        >/dev/null 2>&1 || die "neo $backend $res failed"
    t1=$(now)
    local wall
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
    local bytes frames
    bytes=$(stat -c%s "$out")
    frames=$(count_frames "$out")
    echo "neo_${backend},${res},${run},${wall},${frames},${bytes}" >>"$RESULTS"
    printf "  %-18s %s run=%d  wall=%7ss  frames=%s  bytes=%s\n" \
        "neo_${backend}" "$res" "$run" "$wall" "$frames" "$bytes"
}

run_ffmpeg_nvdec_nvenc() {
    local res=$1 in_mp4=$2 run=$3
    local out="${TMPOUT}.ffmpeg_nvenc.${res}.h264"
    local t0 t1
    t0=$(now)
    ffmpeg -y -loglevel error \
        -hwaccel cuda -hwaccel_output_format cuda \
        -i "$in_mp4" \
        -c:v h264_nvenc -preset p4 -rc constqp -qp 23 \
        -f h264 "$out" \
        >/dev/null 2>&1 || die "ffmpeg nvenc $res failed"
    t1=$(now)
    local wall
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
    local bytes frames
    bytes=$(stat -c%s "$out")
    frames=$(count_frames "$out")
    echo "ffmpeg_nvenc,${res},${run},${wall},${frames},${bytes}" >>"$RESULTS"
    printf "  %-18s %s run=%d  wall=%7ss  frames=%s  bytes=%s\n" \
        "ffmpeg_nvenc" "$res" "$run" "$wall" "$frames" "$bytes"
}

run_ffmpeg_cpu() {
    local res=$1 in_mp4=$2 run=$3
    local out="${TMPOUT}.ffmpeg_libx264.${res}.h264"
    local t0 t1
    t0=$(now)
    ffmpeg -y -loglevel error \
        -i "$in_mp4" \
        -c:v libx264 -preset ultrafast -crf 23 \
        -f h264 "$out" \
        >/dev/null 2>&1 || die "ffmpeg libx264 $res failed"
    t1=$(now)
    local wall
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
    local bytes frames
    bytes=$(stat -c%s "$out")
    frames=$(count_frames "$out")
    echo "ffmpeg_libx264,${res},${run},${wall},${frames},${bytes}" >>"$RESULTS"
    printf "  %-18s %s run=%d  wall=%7ss  frames=%s  bytes=%s\n" \
        "ffmpeg_libx264" "$res" "$run" "$wall" "$frames" "$bytes"
}

warmup() {
    echo "→ warmup: running every config once (discarded)"
    run_neo zerocopy warmup benchmarks/src_1080p.h264 0 >/dev/null 2>&1 || true
    run_ffmpeg_nvdec_nvenc warmup benchmarks/src_1080p.mp4 0 >/dev/null 2>&1 || true
    run_ffmpeg_cpu warmup benchmarks/src_1080p.mp4 0 >/dev/null 2>&1 || true
    # Reset CSV (warmup rows would be there otherwise).
    echo "backend,resolution,run,wall_s,frames,bytes" >"$RESULTS"
}

echo "Neo benchmark — $RUNS runs per config"
echo "============================================="
echo
warmup

for res in 1080p 2160p; do
    echo
    echo "→ resolution: ${res}"
    h264=benchmarks/src_${res}.h264
    mp4=benchmarks/src_${res}.mp4

    for run in $(seq 1 "$RUNS"); do
        echo "  -- run ${run}/${RUNS} --"
        run_neo zerocopy "$res" "$h264" "$run"
        run_neo wgpu     "$res" "$h264" "$run"
        run_neo cpu      "$res" "$h264" "$run"
        run_ffmpeg_nvdec_nvenc "$res" "$mp4" "$run"
        run_ffmpeg_cpu         "$res" "$mp4" "$run"
    done
done

echo
echo "Done. Raw CSV: $RESULTS"
