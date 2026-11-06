@echo off
title Dhan 2021 15-Minute Data Downloader (Secure Private Mode)
cd /d "%~dp0"
cls
echo ======================================================================
echo       DHAN 2021 15-MINUTE INTRADAY DATA DOWNLOADER (PRIVATE)
echo ======================================================================
echo.
echo * Security Note:
echo   Your Client ID and Access Token stay 100%% local in memory.
echo   Nothing is saved to any file, log, or sent to any AI.
echo ======================================================================
echo.
set /p CLIENT_ID="Enter Dhan Client ID: "
echo.
set /p ACCESS_TOKEN="Enter Dhan Access Token: "
echo.
echo ======================================================================
echo [INFO] Starting 2021 download for 30 core stocks via official Dhan API...
echo ======================================================================
echo.
.venv\Scripts\python.exe fetch_dhan_2021.py "%CLIENT_ID%" "%ACCESS_TOKEN%"
echo.
echo ======================================================================
echo Download process finished. Press any key to exit.
echo ======================================================================
pause >nul
