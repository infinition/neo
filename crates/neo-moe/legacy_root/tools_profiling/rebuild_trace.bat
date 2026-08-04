@echo off
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
call "%VS%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
"%CMAKE%" --build "C:\DEV\coding\neo-moe\llama.cpp\bld_cpu_trace" --target llama-neo-moe-trace
exit /b %errorlevel%
