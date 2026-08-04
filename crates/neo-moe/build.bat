@echo off
setlocal enabledelayedexpansion

TITLE neo-moe build

echo ==========================================================
echo  neo-moe : MoE Expert Streaming Engine
echo  Build script
echo ==========================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:: ── Détection de l'OS ────────────────────────────────────────────
if "%OS%"=="Windows_NT" (
    set "TARGET=windows"
    echo [INFO] OS : Windows
) else (
    set "TARGET=linux"
    echo [INFO] OS : Linux
)

:: ── Vérification Rust ─────────────────────────────────────────────
where rustc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] rustc introuvable. Installe Rust : https://rustup.rs
    pause
    exit /b 1
)

:: ── Build ──────────────────────────────────────────────────────────
echo.
echo [BUILD] Compilation de neo-moe en mode release...
echo.

cd /d "%ROOT%"

cargo build --release
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Build echoue.
    pause
    exit /b 1
)

echo.
echo [OK] Build reussi !
echo.
echo  Bibliotheques produites :
if "%TARGET%"=="windows" (
    dir /b "%ROOT%\target\release\neo_moe.dll" 2>nul || echo  neo_moe.dll (cdylib) — utilisee par llama.cpp
) else (
    dir /b "%ROOT%\target\release\libneo_moe.so" 2>nul || echo  libneo_moe.so (cdylib)
)
dir /b "%ROOT%\target\release\neo_moe.lib" 2>nul || echo  neo_moe.lib (import library)
dir /b "%ROOT%\target\release\neo_moe.rlib" 2>nul || echo  neo_moe.rlib (static, Rust consumers)

echo.
echo  Fichier header : %ROOT%\neo_moe_backend.h
echo  Backend C       : %ROOT%\neo_moe_backend.c
echo.
echo  Pour integrer avec llama.cpp :
echo    1. Copier neo_moe.dll / libneo_moe.so dans le dossier des binaires
echo    2. Compiler llama.cpp avec neo_moe_backend.c et -lneo_moe
echo.

endlocal
