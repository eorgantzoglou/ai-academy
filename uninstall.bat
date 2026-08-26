@echo off
setlocal EnableExtensions
title AI Academy - Uninstall

set "DEST=%LOCALAPPDATA%\Programs\AI Academy"

echo.
echo   Removing AI Academy...
echo.

if exist "%DEST%\tools\setup.vbs" (
  cscript //nologo "%DEST%\tools\setup.vbs" remove
) else if exist "%~dp0tools\setup.vbs" (
  cscript //nologo "%~dp0tools\setup.vbs" remove
) else (
  echo   setup.vbs not found - skipping shortcut cleanup.
)

if exist "%DEST%\index.html" (
  REM This script lives inside the folder being deleted, so the delete has
  REM to outlive it: hand the job to a throwaway script in %TEMP%.
  set "CLEAN=%TEMP%\ai-academy-cleanup.bat"
  > "%TEMP%\ai-academy-cleanup.bat" echo @echo off
  >>"%TEMP%\ai-academy-cleanup.bat" echo cd /d "%%TEMP%%"
  >>"%TEMP%\ai-academy-cleanup.bat" echo ping -n 3 127.0.0.1 ^>nul
  >>"%TEMP%\ai-academy-cleanup.bat" echo rd /s /q "%DEST%"
  >>"%TEMP%\ai-academy-cleanup.bat" echo del /q "%%~f0"
  start "" /min cmd /c "%TEMP%\ai-academy-cleanup.bat"
  echo   Program files will be deleted in a moment.
)

echo.
echo   AI Academy has been removed.
echo   Your saved progress lives in the browser and was left alone.
echo.
pause
endlocal
exit /b 0
