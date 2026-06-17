# Runs the yoloworld_live benchmark in both modes and prints a side-by-side summary.
# Run from anywhere; paths resolve against the repo root.
# Usage: .\benchmarks\python\run_compare_yolo.ps1 [-Source assets\videos\bunny.h264] [-Classes "rabbit, butterfly"] [-Frames 50]
param(
    [string]$Source = "assets\videos\bunny.h264",
    [string]$Classes = "rabbit, butterfly",
    [int]$Frames = 50
)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$src_mp4 = [System.IO.Path]::ChangeExtension($Source, ".mp4")

Write-Output "=== Running Neo Mode (Zero-Copy) ==="
& $py "python\examples\yoloworld_live.py" --source $Source --mode neo --classes $Classes --bench $Frames

Write-Output "=== Running Baseline Mode (CPU Decode + Prep) ==="
& $py "python\examples\yoloworld_live.py" --source $src_mp4 --mode baseline --classes $Classes --bench $Frames

$neo = Get-Content "bench_yolo_neo.json" | ConvertFrom-Json
$base = Get-Content "bench_yolo_baseline.json" | ConvertFrom-Json

Write-Output ""
Write-Output "================ NEO vs BASELINE (YOLO-World) ================"
Write-Output ("{0,-28} {1,12} {2,12}" -f "metric", "neo", "baseline")
Write-Output ("{0,-28} {1,12:N2} {2,12:N2}" -f "end-to-end FPS", $neo.fps, $base.fps)
Write-Output ("{0,-28} {1,12:N1} {2,12:N1}" -f "CPU process %", $neo.cpu_process_pct, $base.cpu_process_pct)
Write-Output ("{0,-28} {1,12:N2} {2,12:N2}" -f "host pixel MB/frame", ($neo.host_bytes_per_frame/1e6), ($base.host_bytes_per_frame/1e6))
foreach ($k in $neo.stages_ms.PSObject.Properties.Name) {
    Write-Output ("{0,-28} {1,12:N2} {2,12}" -f "ms: $k (mean)", $neo.stages_ms.$k[0], "-")
}
foreach ($k in $base.stages_ms.PSObject.Properties.Name) {
    Write-Output ("{0,-28} {1,12} {2,12:N2}" -f "ms: $k (mean)", "-", $base.stages_ms.$k[0])
}
