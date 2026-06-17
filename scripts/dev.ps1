# One-shot developer setup: builds the Rust workspace and installs the Python
# bindings into the repo venv.
#
# Usage: .\scripts\dev.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Output "=== cargo build --release (workspace) ==="
cargo build --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path "$root\.venv")) {
    Write-Output "=== creating .venv ==="
    python -m venv "$root\.venv"
}
$py = "$root\.venv\Scripts\python.exe"

Write-Output "=== installing Python deps + neo bindings (maturin develop) ==="
& $py -m pip install --upgrade pip maturin
& $py -m pip install -r "$root\python\requirements.txt"
& $py -m maturin develop --release -m "$root\python\neo-py\Cargo.toml"

Write-Output ""
Write-Output "Ready:"
Write-Output "  .\target\release\neo.exe --help"
Write-Output "  .\target\release\neo-studio.exe"
Write-Output "  .\.venv\Scripts\Activate.ps1 ; python -c `"import neo; print(neo)`""
