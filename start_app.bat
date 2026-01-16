
@echo off
echo Starting Trading App...
start "Backend API" cmd /k "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo Backend started on port 8000
echo Starting Frontend...
cd web
npm run dev
