@echo off
setlocal EnableExtensions
title AI Academy - Install

REM ---------------------------------------------------------------------
REM  Installs AI Academy for the current user. No administrator rights,
REM  no downloads, no runtime to install - everything is copied from the
REM  folder this file sits in.
REM ---------------------------------------------------------------------

set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set "DEST=%LOCALAPPDATA%\Programs\AI Academy"

echo.
echo   ==========================================================
echo      AI Academy
echo      Data Science . ML . Deep Learning . Computer Vision
echo   ==========================================================
echo.

if not exist "%SRC%\index.html" (
  echo   ERROR: index.html was not found next to install.bat.
  echo   Run install.bat from inside the ai-academy folder.
  goto :fail
)
if not exist "%SRC%\tools\setup.vbs" (
  echo   ERROR: tools\setup.vbs is missing - the download is incomplete.
  goto :fail
)

if /I "%SRC%"=="%DEST%" (
  echo   Already installed here. Refreshing shortcuts only.
  goto :shortcuts
)

if exist "%DEST%\index.html" (
  echo   Updating the existing installation.
) else (
  echo   Installing to  %DEST%
)

if not exist "%DEST%" mkdir "%DEST%" 2>nul
if not exist "%DEST%" (
  echo   ERROR: could not create the install folder.
  goto :fail
)

echo   Copying files...
robocopy "%SRC%" "%DEST%" /E /NFL /NDL /NJH /NJS /NP /XD ".git" "node_modules" ".github" /XF "*.lnk" ".gitignore" ".gitattributes" >nul
if errorlevel 8 (
  echo   ERROR: copying the files failed.
  goto :fail
)

:shortcuts
echo   Creating shortcuts...
cscript //nologo "%DEST%\tools\setup.vbs" install "%DEST%"
if errorlevel 1 (
  echo   ERROR: could not create the shortcuts.
  goto :fail
)

echo.
echo   ----------------------------------------------------------
echo    Installed.
echo.
echo      Desktop      AI Academy
echo      Start Menu   AI Academy
echo      Folder       %DEST%
echo      Uninstall    Settings ^> Apps, or uninstall.bat
echo   ----------------------------------------------------------
echo.

choice /C YN /N /M "  Open AI Academy now?  [Y/N] "
if errorlevel 2 goto :done
echo.
echo   Starting...
start "" "%SystemRoot%\System32\wscript.exe" "%DEST%\launch.vbs"

:done
echo.
pause
endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1
