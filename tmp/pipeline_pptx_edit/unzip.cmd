@echo off
if "%~1"=="-Z1" (
  tar.exe -tf "%~2"
  exit /b %errorlevel%
)
if "%~1"=="-p" (
  tar.exe -xOf "%~2" "%~3"
  exit /b %errorlevel%
)
exit /b 2
