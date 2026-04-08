#!/bin/bash
# Start the FastAPI server with Gradio UI mounted on the same port
# HF Spaces only exposes port 7860

set -e

echo "Starting TRACE server (FastAPI + Gradio) on port 7860..."
exec uvicorn server.app:app --host 0.0.0.0 --port 7860
