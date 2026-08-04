@echo off
set "EXE=C:\DEV\coding\neo-moe\llama.cpp\bld_cpu_trace\bin\llama-neo-moe-trace.exe"
set "MODEL=C:\DEV\coding\neo-moe\Qwen_3.6_35b\.models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
set "NEO_TRACE_OUT=C:\DEV\coding\neo-moe\tools_profiling\trace_%1.csv"
"%EXE%" -m "%MODEL%" -ngl 0 -c 4096 -n 800 --no-warmup -p %2
