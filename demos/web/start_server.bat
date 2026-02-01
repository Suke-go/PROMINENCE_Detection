@echo off
echo ========================================
echo  Real-time Prominence Detection Demo
echo ========================================
echo.
echo Starting local server on http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.
echo Opening browser...
start http://localhost:8000/index.html
echo.
python -m http.server 8000
