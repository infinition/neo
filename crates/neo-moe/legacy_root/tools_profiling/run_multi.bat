@echo off
set "EXE=C:\DEV\coding\neo-moe\llama.cpp\bld_cpu_trace\bin\llama-neo-moe-trace.exe"
set "MODEL=C:\DEV\coding\neo-moe\Qwen_3.6_35b\.models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
set "OUT=C:\DEV\coding\neo-moe\tools_profiling"

set "NEO_TRACE_OUT=%OUT%\trace_code.csv"
"%EXE%" -m "%MODEL%" -ngl 0 -c 4096 -n 700 -p "Write a Python class implementing a red-black tree with insert, delete and rebalancing, then explain the rotation logic in detail."

set "NEO_TRACE_OUT=%OUT%\trace_prose.csv"
"%EXE%" -m "%MODEL%" -ngl 0 -c 4096 -n 700 -p "Write a melancholic short story about a lighthouse keeper who finds a message in a bottle from his younger self."

set "NEO_TRACE_OUT=%OUT%\trace_math.csv"
"%EXE%" -m "%MODEL%" -ngl 0 -c 4096 -n 700 -p "Prove rigorously that the square root of 2 is irrational, then compute the integral of x squared times sin x dx step by step."

echo ALL_DONE
exit /b %errorlevel%
