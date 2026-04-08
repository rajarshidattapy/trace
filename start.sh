#!/bin/bash
# Start both the FastAPI backend and Gradio UI

set -e

# Start FastAPI server in background on port 7861
echo "Starting FastAPI backend on port 7861..."
uvicorn server.app:app --host 0.0.0.0 --port 7861 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
sleep 3

# Start Gradio UI on port 7860 (foreground)
echo "Starting Gradio UI on port 7860..."
python -u ui.py
