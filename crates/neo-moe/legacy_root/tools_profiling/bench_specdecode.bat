@echo off
set "BIN=C:\DEV\coding\neo-moe\Qwen_3.6_35b\llama-b9721-bin-win-cuda-12.4-x64\llama-completion.exe"
set "M=C:\DEV\coding\neo-moe\Qwen_3.6_35b\.models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
set "D=C:\DEV\coding\neo-moe\Qwen_3.6_35b\.models\Qwen3-0.6B-Q4_K_M.gguf"
set "LOG=C:\DEV\coding\neo-moe\tools_profiling"
set "P=Explain step by step how a turbojet engine works."

echo [1/2] BASELINE sans draft...
"%BIN%" -m "%M%" -ngl 99 --n-cpu-moe 24 -fa on -c 4096 -n 128 --no-display-prompt ^
  --temp 0.6 --top-p 0.95 --top-k 20 -p "%P%" > "%LOG%\log_baseline.txt" 2>&1

echo [2/2] SPEC-DECODE draft 0.6B...
"%BIN%" -m "%M%" -md "%D%" -ngl 99 --n-cpu-moe 24 -ngld 99 --spec-draft-n-max 4 ^
  -fa on -c 4096 -n 128 --no-display-prompt --temp 0.6 --top-p 0.95 --top-k 20 -p "%P%" > "%LOG%\log_spec.txt" 2>&1

echo ALL_DONE
