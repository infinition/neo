@echo off
setlocal enabledelayedexpansion

TITLE Build llama.cpp + neo-moe

echo ================================================================
echo  Build llama.cpp avec le backend neo-moe
echo  Mode : CPU uniquement (sans CUDA toolkit)
echo ================================================================
echo.

:: ── Chemins ──────────────────────────────────────────────────────────
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "LLAMA_DIR=%ROOT%\llama.cpp"
set "NEO_DIR=%ROOT%\neo-moe"
set "BUILD_DIR=%LLAMA_DIR%\build_neo_moe"

:: ── MSYS2 PATH ───────────────────────────────────────────────────────
set "MSYS2=C:\msys64"
set "UCRT=%MSYS2%\ucrt64\bin"
set "MSYS=%MSYS2%\usr\bin"
set "PATH=%UCRT%;%MSYS%;%PATH%"

:: ── Vérifications ────────────────────────────────────────────────────
if not exist "%LLAMA_DIR%" (
    echo [ERREUR] llama.cpp introuvable. Clone-le d'abord :
    echo   git clone https://github.com/ggerganov/llama.cpp.git
    pause
    exit /b 1
)

if not exist "%NEO_DIR%\target\release\neo_moe.dll" (
    echo [BUILD] Construction de neo-moe...
    cd /d "%NEO_DIR%"
    cargo build --release
    if !ERRORLEVEL! NEQ 0 (
        echo [ERREUR] Build neo-moe echoue.
        pause
        exit /b 1
    )
    echo [OK] neo-moe build termine.
)

:: ── Copie des fichiers backend ───────────────────────────────────────
echo [1/4] Copie des fichiers backend...
copy /Y "%NEO_DIR%\neo_moe_backend.h" "%LLAMA_DIR%\" >nul
copy /Y "%NEO_DIR%\neo_moe_backend.c" "%LLAMA_DIR%\" >nul
echo  [OK] neo_moe_backend.{c,h} copie.

:: ── Configuration CMake ──────────────────────────────────────────────
echo [2/4] Configuration CMake...

if exist "%BUILD_DIR%" rmdir /S /Q "%BUILD_DIR%" 2>nul
mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

cmake .. ^
    -G Ninja ^
    -DCMAKE_C_COMPILER=gcc ^
    -DCMAKE_CXX_COMPILER=g++ ^
    -DGGML_NEO_MOE=ON ^
    -DLLAMA_CUDA=OFF ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_SHARED_LIBS=OFF

if !ERRORLEVEL! NEQ 0 (
    echo [ERREUR] Configuration CMake echouee.
    pause
    exit /b 1
)
echo  [OK] Configuration terminee.

:: ── Compilation ──────────────────────────────────────────────────────
echo [3/4] Compilation...

cmake --build . --target llama-server -- -j%NUMBER_OF_PROCESSORS%
if !ERRORLEVEL! NEQ 0 (
    echo [ERREUR] Compilation echouee.
    pause
    exit /b 1
)
echo  [OK] Compilation terminee.

:: ── Installation ─────────────────────────────────────────────────────
echo [4/4] Installation...

set "LLAMA_BIN=%ROOT%\Qwen_3.6_35b\neo-moe-bin"
if not exist "%LLAMA_BIN%" mkdir "%LLAMA_BIN%"

copy /Y "%BUILD_DIR%\bin\llama-server.exe" "%LLAMA_BIN%\" >nul
copy /Y "%NEO_DIR%\target\release\neo_moe.dll" "%LLAMA_BIN%\" >nul
echo  [OK] Installe dans %LLAMA_BIN%

echo.
echo ================================================================
echo  BUILD REUSSI !
echo ================================================================
echo.
echo  llama-server.exe : %LLAMA_BIN%\llama-server.exe
echo  neo_moe.dll      : %LLAMA_BIN%\neo_moe.dll
echo.
echo  Pour lancer :
echo    cd /d "%ROOT%\Qwen_3.6_35b"
echo    Qwen3.6-35B-A3B-UD-Q4_K_XL.neo-moe.bat
echo.

endlocal
