@echo off
REM Build CPU-only de l'outil de capture de trace (routage MoE identique CPU/GPU).
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "SRC=C:\DEV\coding\neo-moe\llama.cpp"
set "BLD=C:\DEV\coding\neo-moe\llama.cpp\bld_cpu_trace"

call "%VS%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1

"%CMAKE%" -S "%SRC%" -B "%BLD%" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=ON
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%BLD%" --target llama-neo-moe-trace
exit /b %errorlevel%
