# Runs the locate_live benchmark in both modes and prints a side-by-side summary.
# Run from anywhere; paths resolve against the repo root.
# Usage: .\benchmarks\python\run_compare_locate.ps1 [-Source assets\videos\bunny.h264] [-Prompt "the rabbit"] [-Frames 10]
param(
    [string]$Source = "assets\videos\bunny.h264",
    [string]$Prompt = "the rabbit",
    [int]$Frames = 10
)
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$src_mp4 = [System.IO.Path]::ChangeExtension($Source, ".mp4")

& $py "python\examples\locate_live.py" --source $Source --mode neo --prompt $Prompt --bench $Frames
& $py "python\examples\locate_live.py" --source $src_mp4 --mode baseline --prompt $Prompt --bench $Frames

$neo = Get-Content "bench_neo.json" | ConvertFrom-Json
$base = Get-Content "bench_baseline.json" | ConvertFrom-Json

Write-Output ""
Write-Output "================ NEO vs BASELINE ================"
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
