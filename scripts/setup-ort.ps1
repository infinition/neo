# Configures the environment for the ONNX Runtime (CUDA / TensorRT) demos.
#
# Neo loads ONNX Runtime dynamically: it needs a directory containing
# onnxruntime.dll + cuDNN 9 + CUDA 12 runtime DLLs (and optionally TensorRT 10).
# See docs/SETUP.md for the full DLL list and download links.
#
# Usage (per terminal session):
#   .\scripts\setup-ort.ps1 -OrtDir "C:\onnxruntime\lib"
param(
    [string]$OrtDir = "C:\onnxruntime\lib",
    [string]$TrtCacheDir = "$env:LOCALAPPDATA\neo\trt-cache"
)

$dll = Join-Path $OrtDir "onnxruntime.dll"
if (-not (Test-Path $dll)) {
    Write-Error "onnxruntime.dll not found in '$OrtDir'. Download ONNX Runtime GPU 1.22.x for Windows x64 and point -OrtDir at its lib directory (see docs/SETUP.md)."
    exit 1
}

$env:ORT_DYLIB_PATH = $dll
$env:PATH = "$OrtDir;$env:PATH"
New-Item -ItemType Directory -Force -Path $TrtCacheDir | Out-Null
$env:NEO_TRT_CACHE_DIR = $TrtCacheDir

Write-Output "ORT_DYLIB_PATH    = $env:ORT_DYLIB_PATH"
Write-Output "NEO_TRT_CACHE_DIR = $env:NEO_TRT_CACHE_DIR"
Write-Output "PATH              += $OrtDir"
Write-Output ""
Write-Output "Environment ready for this terminal. Set NEO_DISABLE_TRT=1 to force the CUDA EP."
