@echo off
setlocal

TITLE CUDA + neo-moe build

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set "ROOT=%~dp0"
set "BUILD=%ROOT%llama.cpp\bld_cuda2"

if exist "%BUILD%" rmdir /s /q "%BUILD%" 2>nul
mkdir "%BUILD%" 2>nul
cd /d "%BUILD%"

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_BIN=%CUDA_PATH%\bin"
set "PATH=%CUDA_BIN%;%PATH%"
set "CUDACXX=%CUDA_BIN%\nvcc.exe"

echo.
echo [CONFIGURE] CMake with CUDA + neo-moe...
echo.

cmake .. -G Ninja ^
    -DGGML_NEO_MOE=ON ^
    -DGGML_CUDA=ON ^
    -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
    -DCMAKE_CUDA_COMPILER="%CUDACXX%" ^
    -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERREUR] CMake configuration failed.
    pause
    exit /b 1
)

echo.
echo [BUILD] Compilation...
echo.

cmake --build . --target llama-server -- -j%NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERREUR] Build failed.
    pause
    exit /b 1
)

echo.
echo [OK] Build complete!
echo.
echo  Binary: %BUILD%\bin\llama-server.exe
echo.

endlocal
