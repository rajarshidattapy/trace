#!/bin/bash
# Start both the FastAPI backend and Gradio UI

# Start FastAPI server in background on port 7861
echo "Starting FastAPI backend on port 7861..."
uvicorn server.app:app --host 0.0.0.0 --port 7861 &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 2

# Start Gradio UI on port 7860 (foreground)
echo "Starting Gradio UI on port 7860..."
exec python ui.py
