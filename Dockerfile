FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 7860

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run FastAPI backend
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]