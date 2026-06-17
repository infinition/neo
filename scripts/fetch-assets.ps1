# Populates the git-ignored assets/ directory with test clips and weights.
#
#   assets/videos/   test clips (.mp4 + Annex-B .h264)
#   assets/models/   ONNX / .pt model files
#   assets/weights/  raw model weights
#   assets/filters/  drop .onnx files here for `neo filter-live`
#   assets/indexes/  .neo indexes built by the perception-engine example
#
# Requires ffmpeg on PATH for the .mp4 -> Annex-B conversion.
param(
    [switch]$SkipClip,
    [switch]$SkipYolo
)
$root = Split-Path -Parent $PSScriptRoot
foreach ($d in "videos", "models", "weights", "filters", "indexes", "images") {
    New-Item -ItemType Directory -Force -Path (Join-Path $root "assets\$d") | Out-Null
}

if (-not $SkipClip) {
    $mp4 = Join-Path $root "assets\videos\bunny.mp4"
    if (-not (Test-Path $mp4)) {
        Write-Output "Downloading Big Buck Bunny test clip (CC-BY, Blender Foundation)..."
        Invoke-WebRequest "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" -OutFile $mp4
    }
    $h264 = Join-Path $root "assets\videos\bunny.h264"
    if (-not (Test-Path $h264)) {
        Write-Output "Extracting Annex-B H.264 stream..."
        ffmpeg -y -v error -i $mp4 -c:v copy -bsf:v h264_mp4toannexb -an $h264
    }
}

if (-not $SkipYolo) {
    $yolo = Join-Path $root "assets\models\yolov8s-worldv2.pt"
    if (-not (Test-Path $yolo)) {
        Write-Output "Downloading YOLO-World v2 (small) weights..."
        Invoke-WebRequest "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt" -OutFile $yolo
    }
}

Write-Output ""
Write-Output "Done. Other models (RIFE, RealESRGAN, RVM, inswapper) are documented in docs/SETUP.md;"
Write-Output "HuggingFace models (CLIP, LocateAnything) download automatically on first run."
