@echo off
rem Builds ocl-simsand on Windows (run from cmd or PowerShell, or double-click).
rem Usage: windows_build.bat [Debug^|Release]   (default: Debug)
setlocal

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Debug"

rem %~dp0 is this script's folder, with a trailing backslash.
set "BUILD_DIR=%~dp0build"

rem Locate CMake: PATH first, then the default install location.
set "CMAKE=cmake"
where cmake >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files\CMake\bin\cmake.exe" (
        set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
    ) else (
        echo error: CMake not found. Install it or add it to PATH. 1>&2
        exit /b 1
    )
)

rem Configure once; CMake picks the newest installed Visual Studio by default.
rem "%~dp0." rather than "%~dp0": a path ending in \" would escape the closing quote.
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    "%CMAKE%" -S "%~dp0." -B "%BUILD_DIR%" -A x64 || exit /b 1
)

"%CMAKE%" --build "%BUILD_DIR%" --config %CONFIG% --parallel || exit /b 1

echo.
echo Built: %BUILD_DIR%\%CONFIG%\ocl-simsand.exe
