FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port 7860
EXPOSE 7860

# Run FastAPI server (serves HTML UI + REST API)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]