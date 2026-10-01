@echo off
node "%~dp0bin\windows-web.mjs" restart %*
exit /b %errorlevel%
