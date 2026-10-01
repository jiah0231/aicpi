@echo off
node "%~dp0bin\windows-web.mjs" start %*
exit /b %errorlevel%
