@echo off
REM Quick start script for LLM Mini

echo.
echo 🚀 Starting LLM Mini...
echo ====================================
echo GPU-Optimized Language Model
echo AMD RX 7900 XTX Ready
echo ====================================
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    call npm install
)

REM Start the server
echo 🌐 Starting local server on http://localhost:8080
echo 📝 Opening chat interface...
call npm start
